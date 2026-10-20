import { History, FlaskConical, Plus, RotateCcw, Trash2 } from "lucide-react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { useSource } from "../data/context";

export function Shell() {
  const source = useSource();
  const nav = useNavigate();
  const demo = source.kind === "demo";
  return (
    <div className="shell">
      <aside className="sidebar" aria-label="Навигация">
        <div className="brand">
          <span className="brand-mark" aria-hidden>
            <FlaskConical size={20} />
          </span>
          <span>
            <strong>AutoResearch</strong>
            <small>AI-агент для разработки ML-моделей</small>
          </span>
        </div>
        <nav className="nav">
          <NavLink to="/runs" end className="nav-link">
            <History size={18} aria-hidden /> История
          </NavLink>
          <NavLink to="/runs/new" className="nav-link">
            <Plus size={18} aria-hidden /> Новое исследование
          </NavLink>
        </nav>
        {demo && source.demoHistory && (
          <div className="demo-box">
            <div className="demo-title">Управление демо</div>
            {source.demoHistory.isEmpty() ? (
              <button className="btn btn-ghost btn-sm" onClick={() => source.demoHistory!.setEmpty(false)}>
                <RotateCcw size={14} aria-hidden /> Вернуть сценарии
              </button>
            ) : (
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => {
                  source.demoHistory!.setEmpty(true);
                  nav("/runs");
                }}
              >
                <Trash2 size={14} aria-hidden /> Показать пустую историю
              </button>
            )}
          </div>
        )}
      </aside>
      <div className="main">
        <div className="hero-strip" aria-hidden>
          <i className="s1" />
          <i className="s2" />
          <i className="s3" />
        </div>
        {demo && (
          <div className="demo-banner" role="status">
            Демонстрационные данные — Qwen не вызывается, обучение не запускается
          </div>
        )}
        <Outlet />
      </div>
    </div>
  );
}
