import { Navigate, Route, Routes } from "react-router-dom";
import AppShell from "./components/AppShell";
import DemoPage from "./pages/DemoPage";
import HomePage from "./pages/HomePage";
import ModulePlaceholder from "./pages/ModulePlaceholder";
import RunDetailPage from "./pages/RunDetailPage";
import RunsPage from "./pages/RunsPage";

export function AppRouter() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<HomePage />} />
        <Route path="runs" element={<RunsPage />} />
        <Route path="runs/:runId" element={<RunDetailPage />} />
        <Route path="demo" element={<DemoPage />} />
        <Route
          path="modules/curve-fitting"
          element={<ModulePlaceholder moduleKey="curve_fitting" phase={1} />}
        />
        <Route
          path="modules/funclu"
          element={<ModulePlaceholder moduleKey="funclu" phase={2} />}
        />
        <Route
          path="modules/netrecon"
          element={<ModulePlaceholder moduleKey="netrecon" phase={3} />}
        />
        <Route
          path="modules/netanal"
          element={<ModulePlaceholder moduleKey="netanal" phase={5} />}
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
