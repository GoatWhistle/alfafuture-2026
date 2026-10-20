import "@fontsource-variable/manrope";
import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Shell } from "./components/Shell";
import { SourceProvider } from "./data/context";
import { ApiSource } from "./data/api";
import { DemoSource } from "./data/demo";
import { NewRun } from "./screens/NewRun";
import { RunDetail } from "./screens/RunDetail";
import { Runs } from "./screens/Runs";
import "./styles.css";

const source = import.meta.env.VITE_DATA_SOURCE === "api" ? new ApiSource() : new DemoSource();

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <SourceProvider source={source}>
      <BrowserRouter>
        <Routes>
          <Route element={<Shell />}>
            <Route path="/" element={<Navigate to="/runs" replace />} />
            <Route path="/runs" element={<Runs />} />
            <Route path="/runs/new" element={<NewRun />} />
            <Route path="/runs/:id" element={<RunDetail />} />
            <Route path="*" element={<Navigate to="/runs" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </SourceProvider>
  </React.StrictMode>,
);
