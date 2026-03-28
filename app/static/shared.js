const STORAGE_KEY = "petDoctorSession";

export function apiRequest(path, options = {}) {
  const headers = {
    ...(options.body ? { "Content-Type": "application/json" } : {}),
    ...(options.headers || {}),
  };

  return fetch(path, { ...options, headers }).then(async (response) => {
    const rawText = await response.text();
    let data = {};

    if (rawText) {
      try {
        data = JSON.parse(rawText);
      } catch {
        data = { raw: rawText };
      }
    }

    if (!response.ok) {
      throw new Error(
        data.detail || data.message || data.raw || `Request failed: ${response.status}`
      );
    }

    return data;
  });
}

export function serializeForm(form) {
  return Object.fromEntries(
    [...new FormData(form).entries()]
      .map(([key, value]) => [key, typeof value === "string" ? value.trim() : value])
      .filter(([, value]) => value !== "")
  );
}

export function saveSession(user) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify({ user }));
}

export function loadSession() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    return null;
  }

  try {
    return JSON.parse(raw);
  } catch {
    localStorage.removeItem(STORAGE_KEY);
    return null;
  }
}

export function clearSession() {
  localStorage.removeItem(STORAGE_KEY);
}

export function updateSessionUser(user) {
  saveSession(user);
}

export function redirectIfAuthenticated(path = "/chat") {
  const session = loadSession();
  if (session?.user?.id) {
    window.location.href = path;
    return true;
  }
  return false;
}

export function requireSession() {
  const session = loadSession();
  if (!session?.user?.id) {
    window.location.href = "/";
    return null;
  }
  return session;
}

export function setStatus(node, message, tone = "neutral") {
  node.textContent = message;
  node.dataset.tone = tone;
}

export function setButtonBusy(button, busy, busyLabel) {
  if (!button.dataset.defaultLabel) {
    button.dataset.defaultLabel = button.textContent;
  }

  button.disabled = busy;
  button.textContent = busy ? busyLabel : button.dataset.defaultLabel;
}

export function formatValue(value) {
  return value && String(value).trim() ? value : "Not provided";
}

export function clockLabel() {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}
