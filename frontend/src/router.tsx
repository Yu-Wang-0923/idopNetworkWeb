import { Navigate, Route, Routes } from "react-router-dom";
import AppShell from "./components/AppShell";
import CurveFittingPage from "./pages/CurveFittingPage";
import DemoPage from "./pages/DemoPage";
import FunCluPage from "./pages/FunCluPage";
import HomePage from "./pages/HomePage";
import ModulePlaceholder from "./pages/ModulePlaceholder";
import NetReconPage from "./pages/NetReconPage";
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
        <Route path="modules/curve-fitting" element={<CurveFittingPage />} />
        <Route path="modules/funclu" element={<FunCluPage />} />
        <Route path="modules/netrecon" element={<NetReconPage />} />
        <Route
          path="modules/netanal"
          element={<ModulePlaceholder moduleKey="netanal" phase={5} />}
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
