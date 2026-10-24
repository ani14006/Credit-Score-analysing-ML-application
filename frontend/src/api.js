export const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message, status, fieldErrors = {}) {
    super(message);
    this.status = status;
    this.fieldErrors = fieldErrors;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    });
  } catch {
    throw new ApiError(`Cannot reach the backend at ${API_URL}. Is it running?`, 0);
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body?.detail;
    if (response.status === 422 && Array.isArray(detail)) {
      const fieldErrors = {};
      for (const err of detail) fieldErrors[err.loc[err.loc.length - 1]] = err.msg;
      throw new ApiError("The backend rejected some inputs.", 422, fieldErrors);
    }
    throw new ApiError(typeof detail === "string" ? detail : `Request failed (${response.status})`, response.status);
  }
  return body;
}

export const api = {
  health: () => request("/health"),
  metrics: () => request("/metrics"),
  modelInfo: () => request("/model-info"),
  schema: () => request("/openapi.json").then((s) => s.components.schemas.CustomerFeatures),
  predict: (features) => request("/predict", { method: "POST", body: JSON.stringify(features) }),
  predictions: (limit = 10) => request(`/predictions?limit=${limit}`),
};
