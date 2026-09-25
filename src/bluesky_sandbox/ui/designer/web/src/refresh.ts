// The Refresh button's count. Every effect that fetches from the backend lists
// it as a dependency, so a refresh re-runs them in place - the tabs stay
// mounted, keeping the map's view and the editors' state.
import { createContext, useContext } from "react";

export const RefreshContext = createContext(0);

export const useRefresh = () => useContext(RefreshContext);
