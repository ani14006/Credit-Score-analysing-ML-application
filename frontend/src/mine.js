// Per-browser record of this user's own predictions: { id, token, sessionId? }.
// The token is the capability the backend issued for that prediction; without it the
// prediction and its agent session cannot be read. Storage can be unavailable (private mode),
// so every access is guarded and the app still works for the current page view.
const KEY = "credit-risk.my-predictions";
const MAX = 50;

function read() {
  try {
    const items = JSON.parse(localStorage.getItem(KEY) || "[]");
    return Array.isArray(items) ? items.filter((i) => Number.isInteger(i.id) && typeof i.token === "string") : [];
  } catch {
    return [];
  }
}

function write(items) {
  try {
    localStorage.setItem(KEY, JSON.stringify(items.slice(0, MAX)));
  } catch {
    /* storage unavailable: keep working without persistence */
  }
}

export const mine = {
  all: read,
  add(id, token) {
    write([{ id, token }, ...read().filter((i) => i.id !== id)]);
  },
  token: (id) => read().find((i) => i.id === id)?.token ?? null,
  sessionId: (id) => read().find((i) => i.id === id)?.sessionId ?? null,
  setSession(id, sessionId) {
    write(read().map((i) => (i.id === id ? { ...i, sessionId } : i)));
  },
};
