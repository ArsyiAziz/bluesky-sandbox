# Rendering

You can switch renderers with a single argument. By default `render_mode=None` runs headless, which is what you
want during training.

| Mode | Visual | Best for |
| :--- | :--- | :--- |
| `pygame` | ![Pygame](media/screenshots/pygame.png) | High-speed 2D prototyping and quick debugging |
| `panda3d` | ![Panda3D](media/screenshots/panda3d.png) | 3D altitude visualization and spatial analysis |
| `qtgl` | ![QtGL](media/screenshots/qtgl.png) | High-fidelity BlueSky native radar display |

```python
env = Env(render_mode="pygame")
```

Each renderer contains optional extras — see [Installation](installation.md#optional-extras).

## Real time and views

`realtime=True` runs the simulation following wall-clock time.

The `views` argument controls the panel layout. See
[Rendering and drivers](api/rendering.md) for the driver and view types.
