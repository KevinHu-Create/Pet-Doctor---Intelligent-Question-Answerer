import {
  apiRequest,
  clearSession,
  formatValue,
  requireSession,
  setStatus,
  updateSessionUser,
} from "/static/shared.js";

const session = requireSession();

if (session) {
  const backToChat = document.getElementById("back-to-chat");
  const logoutButton = document.getElementById("profile-logout");
  const profileStatus = document.getElementById("profile-status");
  const form = document.getElementById("profile-form");
  const editButton = document.getElementById("edit-profile");
  const saveButton = document.getElementById("save-profile");
  const cancelButton = document.getElementById("cancel-profile");
  const deleteButton = document.getElementById("delete-profile");

  const fields = {
    id: form.elements.id,
    username: form.elements.username,
    email: form.elements.email,
    role: form.elements.role,
    pet_name: form.elements.pet_name,
    pet_type: form.elements.pet_type,
    password: form.elements.password,
  };

  const summaryNodes = {
    username: document.getElementById("summary-username"),
    email: document.getElementById("summary-email"),
    role: document.getElementById("summary-role"),
    petName: document.getElementById("summary-pet-name"),
    petType: document.getElementById("summary-pet-type"),
  };

  let currentUser = null;

  function setEditing(editing) {
    ["username", "email", "pet_name", "pet_type", "password"].forEach((key) => {
      fields[key].disabled = !editing;
    });
    fields.id.disabled = true;
    fields.role.disabled = true;

    editButton.classList.toggle("is-hidden", editing);
    saveButton.classList.toggle("is-hidden", !editing);
    cancelButton.classList.toggle("is-hidden", !editing);

    if (!editing) {
      fields.password.value = "";
    }
  }

  function renderUser(user) {
    fields.id.value = user.id ?? "";
    fields.username.value = user.username ?? "";
    fields.email.value = user.email ?? "";
    fields.role.value = user.role ?? session.user.role ?? "user";
    fields.pet_name.value = user.pet_name ?? "";
    fields.pet_type.value = user.pet_type ?? "";
    fields.password.value = "";

    summaryNodes.username.textContent = formatValue(user.username);
    summaryNodes.email.textContent = formatValue(user.email);
    summaryNodes.role.textContent = formatValue(user.role || session.user.role || "user");
    summaryNodes.petName.textContent = formatValue(user.pet_name);
    summaryNodes.petType.textContent = formatValue(user.pet_type);
  }

  async function loadUser() {
    setStatus(profileStatus, "Loading current user profile.", "neutral");

    try {
      const user = await apiRequest(`/users/${session.user.id}`);
      currentUser = user;
      renderUser(user);
      updateSessionUser({ ...session.user, ...user });
      setStatus(profileStatus, "Profile loaded.", "success");
    } catch (error) {
      setStatus(profileStatus, error.message, "error");
    }
  }

  backToChat.addEventListener("click", () => {
    window.location.href = "/chat";
  });

  logoutButton.addEventListener("click", () => {
    clearSession();
    window.location.href = "/";
  });

  editButton.addEventListener("click", () => {
    setEditing(true);
    setStatus(profileStatus, "Edit mode enabled.", "neutral");
  });

  cancelButton.addEventListener("click", () => {
    if (currentUser) {
      renderUser(currentUser);
    }
    setEditing(false);
    setStatus(profileStatus, "Edits discarded.", "neutral");
  });

  saveButton.addEventListener("click", async () => {
    const payload = {
      username: fields.username.value.trim(),
      email: fields.email.value.trim(),
      pet_name: fields.pet_name.value.trim(),
      pet_type: fields.pet_type.value.trim(),
    };

    if (fields.password.value.trim()) {
      payload.password = fields.password.value.trim();
    }

    setStatus(profileStatus, "Saving updated profile.", "neutral");
    saveButton.disabled = true;

    try {
      const user = await apiRequest(`/users/${session.user.id}`, {
        method: "PUT",
        body: JSON.stringify(payload),
      });
      currentUser = user;
      renderUser(user);
      updateSessionUser({ ...session.user, ...user });
      setEditing(false);
      setStatus(profileStatus, "Profile updated successfully.", "success");
    } catch (error) {
      setStatus(profileStatus, error.message, "error");
    } finally {
      saveButton.disabled = false;
    }
  });

  deleteButton.addEventListener("click", async () => {
    const confirmed = window.confirm(
      "Delete this account? This action cannot be undone."
    );

    if (!confirmed) {
      return;
    }

    deleteButton.disabled = true;
    setStatus(profileStatus, "Deleting account.", "neutral");

    try {
      await apiRequest(`/users/${session.user.id}`, {
        method: "DELETE",
      });
      clearSession();
      window.location.href = "/";
    } catch (error) {
      setStatus(profileStatus, error.message, "error");
      deleteButton.disabled = false;
    }
  });

  setEditing(false);
  loadUser();
}
