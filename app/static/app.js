const apiResult = document.getElementById("api-result");
const askResult = document.getElementById("ask-result");
const apiStatusText = document.getElementById("api-status-text");

const sampleQuestion =
  "My dog has mild diarrhea but still seems active. What should I pay attention to?";

function toObject(formData) {
  return Object.fromEntries(
    [...formData.entries()]
      .map(([key, value]) => [key, typeof value === "string" ? value.trim() : value])
      .filter(([, value]) => value !== "")
  );
}

function printResult(title, payload) {
  apiResult.textContent = `${title}\n\n${JSON.stringify(payload, null, 2)}`;
}

function setStatus(text) {
  apiStatusText.textContent = text;
}

async function request(path, options = {}) {
    setStatus(`Calling ${path}...`);
    try {
    const headers = {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    };

      const response = await fetch(path, {
        ...options,
        headers,
      });

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
          data.detail || data.raw || `Request failed: ${response.status}`
        );
      }

      setStatus(`Last request succeeded: ${path}`);
      return data;
    } catch (error) {
      setStatus(`Last request failed: ${path}`);
      throw error;
    }
  }

async function handleFormSubmit(formId, path, options = {}) {
  const form = document.getElementById(formId);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = toObject(new FormData(form));

    try {
      const data = await request(path(payload), {
        method: options.method || "POST",
        body: options.body ? options.body(payload) : JSON.stringify(payload),
      });
      printResult(`${options.title || formId} success`, data);

      if (options.onSuccess) {
        options.onSuccess(data);
      }
    } catch (error) {
      printResult(`${options.title || formId} error`, { error: error.message });
    }
  });
}

document.querySelector("[data-fill-question]").addEventListener("click", () => {
  document.getElementById("question").value = sampleQuestion;
});

handleFormSubmit("ask-form", () => "/ask", {
  title: "Ask Pet Doctor",
  onSuccess(data) {
    askResult.textContent = data.answer || "No answer returned.";
  },
});

handleFormSubmit("register-form", () => "/register", {
  title: "Register",
  body(payload) {
    return JSON.stringify({
      ...payload,
      role: payload.role || "user",
    });
  },
});

handleFormSubmit("login-form", () => "/login", {
  title: "Login",
});

document.getElementById("list-users-btn").addEventListener("click", async () => {
  try {
    const data = await request("/users/");
    printResult("List Users success", data);
  } catch (error) {
    printResult("List Users error", { error: error.message });
  }
});

handleFormSubmit("get-user-form", (payload) => `/users/${payload.user_id}`, {
  title: "Get User",
  method: "GET",
  body: null,
});

handleFormSubmit("create-user-form", () => "/users/", {
  title: "Create User",
});

handleFormSubmit("update-user-form", (payload) => `/users/${payload.user_id}`, {
  title: "Update User",
  method: "PUT",
  body(payload) {
    const { user_id, ...rest } = payload;
    return JSON.stringify(rest);
  },
});

handleFormSubmit("delete-user-form", (payload) => `/users/${payload.user_id}`, {
  title: "Delete User",
  method: "DELETE",
  body: () => null,
});

async function warmup() {
  try {
    const data = await request("/health");
    printResult("Health check", data);
  } catch (error) {
    printResult("Health check error", { error: error.message });
  }
}

warmup();
