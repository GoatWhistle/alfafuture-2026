import { useEffect, useRef } from "react";
import type { ReactNode } from "react";

interface Props {
  title: string;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
  side?: boolean;
}

/** Диалог с возвратом фокуса, закрытием по Esc и удержанием Tab внутри */
export function Modal({ title, onClose, children, footer, side }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null;
    const root = ref.current!;
    const focusables = () =>
      Array.from(root.querySelectorAll<HTMLElement>("button, [href], input, select, textarea, summary, [tabindex]:not([tabindex='-1'])"));
    (focusables()[0] ?? root).focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
      } else if (e.key === "Tab") {
        const f = focusables();
        if (!f.length) return;
        const first = f[0];
        const last = f[f.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    root.addEventListener("keydown", onKey);
    return () => {
      root.removeEventListener("keydown", onKey);
      prev?.focus();
    };
  }, [onClose]);

  return (
    <div className={`overlay ${side ? "overlay-side" : ""}`} onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div ref={ref} className={side ? "drawer" : "dialog"} role="dialog" aria-modal="true" aria-label={title} tabIndex={-1}>
        {children}
        {footer && <div className="dialog-footer">{footer}</div>}
      </div>
    </div>
  );
}
