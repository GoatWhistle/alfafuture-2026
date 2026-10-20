import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { DataSource } from "./source";

const Ctx = createContext<DataSource | null>(null);

export function SourceProvider({ source, children }: { source: DataSource; children: ReactNode }) {
  return <Ctx.Provider value={source}>{children}</Ctx.Provider>;
}

export function useSource(): DataSource {
  const s = useContext(Ctx);
  if (!s) throw new Error("SourceProvider отсутствует");
  return s;
}

export interface Loaded<T> {
  data: T | undefined;
  loading: boolean;
  error: string | null;
  reload: () => void;
}

/**
 * Загружает данные и обновляет их при уведомлении источника.
 * Опрос выполняется только пока `active` истинно, затем прекращается.
 */
export function useLoaded<T>(load: (s: DataSource) => Promise<T>, deps: unknown[], active = false): Loaded<T> {
  const source = useSource();
  const [data, setData] = useState<T>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const loadRef = useRef(load);
  loadRef.current = load;

  const reload = useCallback(() => {
    loadRef
      .current(source)
      .then((d) => {
        setData(d);
        setError(null);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source, ...deps]);

  useEffect(() => {
    setLoading(true);
    reload();
    return source.subscribe(reload);
  }, [reload, source]);

  useEffect(() => {
    if (!active || source.pollMs === null) return;
    const t = setInterval(reload, source.pollMs);
    return () => clearInterval(t);
  }, [active, reload, source]);

  return { data, loading, error, reload };
}
