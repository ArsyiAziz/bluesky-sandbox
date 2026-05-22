"""QtGL sim driver - drives the BlueSky QtGL window for mode='sim' environments.

How it works
------------
BlueSky's ``mode='sim'`` keeps the full ``bs_traf`` object in-process, but its
``ScreenIO`` layer already publishes live traffic data (ACDATA, SIMINFO, TRAILS...)
over ZMQ at 5 Hz - it just expects a server to route the messages.

This module provides:
- ``QtGLSimDriver``: extends ``HumanSimDriver`` by starting a ZMQ proxy thread
  and spawning a ``python -m bluesky --client localhost`` subprocess that opens
  the full BlueSky QtGL radar window.  Overrides ``update()`` to flush traffic
  data over ZMQ and block on GUI pause/hold.

The RL environment retains direct ``bs_traf`` access; the QtGL window receives
all state updates via the normal BlueSky network protocol.

Node registration and act_id
-----------------------------
The GUI's status bar (SIMINFO: clock, speed, UTC ...) only updates for the
*active node* - the node whose ID matches ``bs.net.act_id`` inside the GUI
process.  The GUI sets ``act_id`` via its ``node_added`` signal, which fires
when the GUI's ``sock_send`` (XPUB) receives the sim node's ZMQ registration
subscription message.

That registration message is sent exactly once when ``bs.net.connect()`` is
called.  If the GUI's sock_send hasn't connected to the proxy yet at that
moment, it misses the message and ``act_id`` stays ``None`` forever.

To work around this we:
1. Launch the GUI *before* calling ``bs.net.connect()``.
2. Sleep long enough for the Qt process to start and bind its sockets.
3. In ``update()`` we also re-announce the sim node every few calls for the
   first ~30 seconds so slow GUI starts are handled automatically.
"""

from __future__ import annotations

import math
import subprocess
import sys
import threading
import time

import bluesky as bs
import zmq

from bluesky_sandbox.ui.display.overlays import Point, Polygon, Polyline
from bluesky_sandbox.ui.drivers.human_driver import HumanSimDriver


class QtGLSimDriver(HumanSimDriver):
    """Drives the BlueSky QtGL window alongside a ``mode='sim'`` environment.

    Extends :class:`HumanSimDriver` by starting a ZMQ proxy thread and spawning
    the QtGL client subprocess.  Overrides ``update()`` to flush traffic state
    over ZMQ and block when the user pauses the simulation from the GUI.

    Parameters
    ----------
    realtime:
        Passed to :class:`SimDriver`.  Defaults to ``True`` so the window runs
        at real-time speed by default.
    """

    # Number of update() calls over which to keep re-announcing the sim node.
    _REG_WINDOW = 150

    # Radius in nautical miles for the CIRCLE used to render Point primitives.
    # BlueSky's CIRCLE radius is in absolute nm, so the visual size depends on
    # zoom level.
    _POINT_RADIUS_NM = 1

    # Number of concentric CIRCLEs stacked to fake a solid filled disk in the
    # point's color.  BlueSky's SHOWPOLY 2 only fills with the palette default
    # (translucent green) - per-shape COLOR controls the outline only - so we
    # nest outlines at decreasing radii until the disk reads as solid.
    _POINT_RING_COUNT = 10

    def __init__(self, realtime: bool = True) -> None:
        super().__init__(realtime=realtime)
        self._proxy: _ZMQProxy | None = None
        self._proc:  subprocess.Popen | None = None
        self._reg_count: int = 0   # counts update() calls for re-announcement
        # Per-aircraft COLOR last sent to the GUI - avoids re-issuing the
        # same command every substep when conflict state hasn't changed.
        self._last_colors: dict[str, str] = {}
        # Names of the defined-route POLYLINE shapes currently drawn, so the
        # "show all routes" toggle can remove them again.
        self._route_shape_names: list[str] = []

    def toggle_trails(self) -> None:
        """Flip :attr:`show_trails` and forward to BlueSky's TRAIL command.

        The QtGL window has its own trail layer driven by the BlueSky
        ``TRAIL`` stack command; stacking ``ON`` / ``OFF`` keeps the
        GUI's state in sync with the driver flag exposed on
        :class:`HumanSimDriver`.
        """
        super().toggle_trails()
        bs.stack.stack(f"TRAIL {'ON' if self.show_trails else 'OFF'}")

    def start(self) -> None:
        """Bind the ZMQ broker, connect the sim node, and open the QtGL window."""
        super().start()
        # 1. Start the in-process ZMQ proxy (message broker).
        self._proxy = _ZMQProxy()
        self._proxy.start()

        # 2. Launch the QtGL client BEFORE connecting the sim node.
        #    See module docstring for why the order matters.
        #    stderr is redirected to DEVNULL so Qt/ZMQ teardown messages from
        #    the subprocess don't appear in the user's terminal on close.
        print("BlueSky: launching QtGL window...", flush=True)
        self._proc = subprocess.Popen(
            [sys.executable, '-m', 'bluesky', '--client', 'localhost'],
            stderr=subprocess.DEVNULL,
        )

        # 3. Connect the sim node so the (already-connected) GUI sees its
        #    registration subscription and sets act_id.
        bs.net.connect(hostname='localhost')
        print("BlueSky: sim node connected to QtGL client.", flush=True)

    def wait_until_ready(self, timeout: float = 30.0) -> None:
        """Block until the QtGL GUI has connected to the ZMQ broker.

        Raises ``TimeoutError`` if the GUI does not connect within *timeout* seconds.
        """
        if not self._proxy.gui_connected.wait(timeout=timeout):
            raise TimeoutError(
                f"BlueSky QtGL window did not connect within {timeout}s"
            )

    def update(self) -> None:
        """Send traffic state to the GUI and process incoming commands.

        Raises ``SystemExit`` if the QtGL window has been closed, so that a
        normal ``while True`` training loop terminates cleanly.

        Mirrors the BlueSky run loop order:
        1. ``Timer.update_timers()`` fires the wall-clock timers that trigger
           ``send_aircraft_data()`` (ACDATA) and ``send_siminfo()`` (SIMINFO).
           Without this, those callbacks never fire and the GUI receives nothing.
        2. ``bs.net.update()`` flushes queued ZMQ messages and receives any
           incoming commands from the GUI client.
        3. ``bs.scr.update()`` increments the sample counter used by SIMINFO.
        4. For the first _REG_WINDOW calls, re-announce the sim node every
           10 calls.  This handles GUI processes that start slower than the
           initial 5 s sleep and ensures act_id is always set eventually.
        """
        if self._proc is not None and self._proc.poll() is not None:
            raise SystemExit("BlueSky GUI window closed")

        from bluesky.core.walltime import Timer
        from bluesky.stack import simstack

        Timer.update_timers()
        bs.net.update()
        bs.scr.update()
        simstack.process()

        self._reg_count += 1
        if self._reg_count <= self._REG_WINDOW and self._reg_count % 10 == 0:
            # Cycle unsubscribe -> subscribe to emit a fresh registration
            # message through the proxy so the GUI can detect this node.
            bs.net.unsubscribe('', '', to_group=bs.net.node_id)
            bs.net.subscribe('', '', to_group=bs.net.node_id)

        # If the GUI sent a HOLD command, block here until OP is received.
        # bs.sim.state is set to bs.HOLD by bs.sim.hold() and to bs.OP by bs.sim.op().
        # We must call simstack.process() so that the incoming OP command is
        # actually executed and updates bs.sim.state - without it, the command
        # sits queued in the stack forever and the loop never exits.
        if bs.sim.state == bs.HOLD:
            while bs.sim.state == bs.HOLD:
                if self._proc is not None and self._proc.poll() is not None:
                    raise SystemExit("BlueSky GUI window closed")
                Timer.update_timers()
                bs.net.update()
                bs.scr.update()
                simstack.process()
                time.sleep(0.05)

    def on_reset(self, env=None) -> None:
        """Draw all renderables and pan to fit after a reset."""
        # Aircraft are recreated on reset - flush the color cache so the
        # first update() re-sends COLOR for every fresh callsign.
        self._last_colors.clear()
        # bs.sim.reset() clears drawn shapes, so the previous routes are gone.
        self._route_shape_names = []
        super().on_reset(env)
        if self._env is None:
            raise RuntimeError("QtGLSimDriver env has not been bound.")
        if self.show_all_routes:
            self._draw_defined_routes()
        self._pan_to_center(
            self._env.episode_spawn.resolved_bounds,
            self._env.episode_airspace_bounds,
        )

    def toggle_all_routes(self) -> None:
        """Flip the flag and add/remove the defined-route polylines live."""
        from bluesky.stack import simstack

        super().toggle_all_routes()
        if self.show_all_routes:
            self._draw_defined_routes()
        else:
            self._clear_defined_routes()
            simstack.process()

    def _draw_defined_routes(self) -> None:
        """Stack ``POLYLINE``/``COLOR`` for each of the design's defined routes."""
        from bluesky.stack import simstack

        self._clear_defined_routes()
        for i, pts in enumerate(self.defined_route_polylines()):
            name = f"ROUTE_{i}"
            coords = " ".join(f"{lat},{lon}" for lat, lon, _ in pts)
            bs.stack.stack(f"POLYLINE {name} {coords}")
            bs.stack.stack(f"COLOR {name} cyan")
            self._route_shape_names.append(name)
        simstack.process()

    def _clear_defined_routes(self) -> None:
        """Delete any previously drawn defined-route polylines."""
        for name in self._route_shape_names:
            bs.stack.stack(f"DELPOLY {name}")
        self._route_shape_names = []

    def on_render(self, env=None) -> None:
        """Pan the view to fit and switch polygons to filled mode.

        ``SHOWPOLY 2`` toggles BlueSky's GUI between off / outline /
        outline+fill - mode 2 fills every poly and circle with a
        translucent tint (palette alpha ~50).  This is what makes the
        :meth:`draw_point` CIRCLE markers read as solid dots instead of
        rings; airspace and query regions also pick up the same
        translucent fill, which is the look most BlueSky users expect.
        """
        super().on_render(env)
        if self._env is None:
            raise RuntimeError("QtGLSimDriver env has not been bound.")
        bs.stack.stack("SHOWPOLY 2")
        self._pan_to_center(
            self._env.episode_spawn.resolved_bounds,
            self._env.episode_airspace_bounds,
        )

    def draw_polygon(self, polygon: Polygon) -> None:
        """Stack ``POLY name ...; COLOR name ...`` for the given polygon."""
        name = self._stack_name(polygon.label, prefix="POLY", obj=polygon)
        verts = " ".join(f"{lat},{lon}" for lat, lon in polygon.vertices)
        bs.stack.stack(f"POLY {name} {verts}")
        bs.stack.stack(f"COLOR {name} {polygon.color}")

    def draw_point(self, point: Point) -> None:
        """Stack concentric ``CIRCLE``s to fake a solid marker in the point's color.

        BlueSky has no native filled-point primitive: ``SHOWPOLY 2`` only
        fills with the global palette default, not per-shape ``COLOR``.
        Stacking :attr:`_POINT_RING_COUNT` rings at evenly-decreasing
        radii nests their outlines tightly enough that the disk reads as
        solid in the point's color at typical zoom levels.

        ``alt_ft`` is intentionally ignored: BlueSky's CIRCLE alt range
        controls area-inclusion semantics, not the visual.
        """
        base = self._stack_name(point.label, prefix="POINT", obj=point)
        n = self._POINT_RING_COUNT
        for i in range(n):
            r = self._POINT_RADIUS_NM * (n - i) / n
            ring = f"{base}_R{i}"
            bs.stack.stack(f"CIRCLE {ring} {point.lat} {point.lon} {r}")
            bs.stack.stack(f"COLOR {ring} {point.color}")

    def draw_polyline(self, polyline: Polyline) -> None:
        """Stack ``POLYLINE name ...; COLOR name ...`` for the given polyline."""
        name = self._stack_name(polyline.label, prefix="LINE", obj=polyline)
        pts = " ".join(f"{lat},{lon}" for lat, lon in polyline.points)
        bs.stack.stack(f"POLYLINE {name} {pts}")
        bs.stack.stack(f"COLOR {name} {polyline.color}")

    @staticmethod
    def _stack_name(label: str, prefix: str, obj: object) -> str:
        """Build a single-token shape ID for BlueSky's stack commands.

        BlueSky shape commands (``POLY``, ``CIRCLE``, ``POLYLINE``) take
        a single-token name, so we replace spaces in the label.  When no
        label is set, we derive a stable id from ``meta["bounds"]`` -
        ``BoundsResource`` always stashes the underlying ``Bounds`` there,
        and bounds outlive a single ``render_primitives()`` call.  Falling
        back to ``id(obj)`` here would alias two anonymous primitives
        whose temporary ``Polygon`` instances happen to land at the same
        CPython address after the previous one was GC'd - silently
        overwriting one shape with the next on the GUI side.
        """
        if label:
            return label.replace(" ", "_")
        meta = getattr(obj, "meta", None)
        bounds = meta.get("bounds") if isinstance(meta, dict) else None
        if bounds is not None:
            return f"{prefix}{id(bounds):x}"
        return f"{prefix}{id(obj):x}"

    def _pan_to(self, lat: float, lon: float) -> None:
        """Pan the radar view to the given lat/lon."""
        bs.stack.stack(f"PAN {lat} {lon}")

    def _pan_to_center(self, bounds: dict, spatial_bounds=None, margin: float = 1.1) -> None:
        """Pan and zoom the radar view to fit the given spawn bounds.

        Uses the same zoom formula as BlueSky's built-in area list:
        ``zoom = 1 / max(lat_range, lon_range * cos(center_lat))``,
        multiplied by ``1/margin`` so the box has breathing room.

        Parameters
        ----------
        bounds:
            Dict with ``"lat_deg"`` and ``"lon_deg"`` keys, each a ``(min, max)`` tuple.
        spatial_bounds:
            Optional :class:`~bluesky_sandbox.sim.bounds.Bounds` instance.
            When provided its bounding box is used for the pan/zoom calculation.
        margin:
            Scale factor > 1 that adds padding around the box (default 1.1 = 10%).
        """
        if spatial_bounds is not None:
            (lat_min, lat_max), (lon_min, lon_max) = spatial_bounds.bounding_box
        else:
            lat_min, lat_max = bounds.get('lat_deg', (-90, 90))
            lon_min, lon_max = bounds.get('lon_deg', (-180, 180))
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        lat_range = lat_max - lat_min
        lon_range = (lon_max - lon_min) * math.cos(math.radians(center_lat))
        zoom = 1.0 / (max(lat_range, lon_range) * margin)
        bs.stack.stack(f"PAN {center_lat} {center_lon};ZOOM {zoom}")

    def close(self) -> None:
        """Terminate the QtGL subprocess and stop the ZMQ proxy cleanly."""
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait(timeout=3)
            self._proc = None
        if self._proxy is not None:
            self._proxy.stop()
            self._proxy = None


class _ZMQProxy(threading.Thread):
    """Minimal ZMQ XSUB/XPUB message broker.

    Binds to the same ports that BlueSky's Server uses so that both the sim
    node and the QtGL client can connect without modification.

    Port mapping (from BlueSky's defaults):
        recv_port = 11000 -> XPUB  (clients/nodes subscribe here)
        send_port = 11001 -> XSUB  (nodes publish here)

    The broker forwards all messages between the two sockets via a manual
    poll loop.  This allows us to inspect subscription messages from the GUI
    and set ``gui_connected`` as soon as the QtGL client has registered.
    """

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.name = 'BlueSkyZMQProxy'
        self.gui_connected = threading.Event()
        self._stop = threading.Event()

    def stop(self) -> None:
        """Signal the poll loop to exit."""
        self._stop.set()

    def run(self) -> None:
        ctx = zmq.Context.instance()

        # Nodes' XPUB sockets connect to send_port -> we bind XSUB here
        xsub = ctx.socket(zmq.XSUB)
        xsub.bind(f'tcp://*:{bs.settings.send_port}')  # 11001

        # Nodes' SUB sockets connect to recv_port -> we bind XPUB here
        xpub = ctx.socket(zmq.XPUB)
        xpub.bind(f'tcp://*:{bs.settings.recv_port}')  # 11000

        poller = zmq.Poller()
        poller.register(xsub, zmq.POLLIN)
        poller.register(xpub, zmq.POLLIN)

        try:
            while not self._stop.is_set():
                ready = dict(poller.poll(timeout=100))

                # Subscription messages arrive on XPUB (from GUI subscribers).
                # Forward them to XSUB so publishers know who is listening.
                # A leading b'\x01' byte means "subscribe" - the GUI just connected.
                if xpub in ready:
                    while True:
                        try:
                            msg = xpub.recv_multipart(zmq.NOBLOCK)
                            if msg[0][:1] == b'\x01':
                                self.gui_connected.set()
                            xsub.send_multipart(msg)
                        except zmq.Again:
                            break

                # Data messages arrive on XSUB (from the sim node).
                # Forward them to XPUB so all subscribers receive them.
                if xsub in ready:
                    while True:
                        try:
                            xpub.send_multipart(xsub.recv_multipart(zmq.NOBLOCK))
                        except zmq.Again:
                            break
        except zmq.ZMQError:
            pass
        finally:
            xsub.close()
            xpub.close()
