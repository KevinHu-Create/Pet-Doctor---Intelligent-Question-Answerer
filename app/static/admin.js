import {
  apiRequest,
  clearSession,
  formatValue,
  requireSession,
  setStatus,
} from "/static/shared.js";

const session = await requireSession("admin");

if (session) {
  const logoutButton = document.getElementById("admin-logout");
  const refreshButton = document.getElementById("admin-refresh");
  const adminStatus = document.getElementById("admin-status");
  const userCount = document.getElementById("user-count");
  const activeCount = document.getElementById("active-count");
  const tbody = document.getElementById("admin-users-body");
  const adminBadge = document.getElementById("admin-badge");

  adminBadge.textContent = `${session.user.username} • Administrator`;

  function renderUsers(users) {
    tbody.innerHTML = "";
    userCount.textContent = String(users.length);
    activeCount.textContent = String(users.filter((user) => user.is_active !== false).length);

    if (!users.length) {
      const row = document.createElement("tr");
      row.innerHTML = '<td colspan="6" class="table-empty">No registered users yet.</td>';
      tbody.append(row);
      return;
    }

    users.forEach((user) => {
      const row = document.createElement("tr");
      const petLabel =
        user.pet_name || user.pet_type
          ? `${formatValue(user.pet_name)} / ${formatValue(user.pet_type)}`
          : "Not provided";

      row.innerHTML = `
        <td>${user.id}</td>
        <td>${formatValue(user.username)}</td>
        <td>${formatValue(user.email)}</td>
        <td>${petLabel}</td>
        <td><span class="table-role">${formatValue(user.role)}</span></td>
        <td><button type="button" class="danger-button admin-delete" data-user-id="${user.id}">Delete</button></td>
      `;
      tbody.append(row);
    });
  }

  async function loadUsers() {
    setStatus(adminStatus, "Loading registered users.", "neutral");
    refreshButton.disabled = true;

    try {
      const users = await apiRequest("/users/");
      renderUsers(users);
      setStatus(adminStatus, "User list updated.", "success");
    } catch (error) {
      setStatus(adminStatus, error.message, "error");
    } finally {
      refreshButton.disabled = false;
    }
  }

  tbody.addEventListener("click", async (event) => {
    const button = event.target.closest(".admin-delete");
    if (!button) {
      return;
    }

    const { userId } = button.dataset;
    const confirmed = window.confirm(`Delete user #${userId}? This cannot be undone.`);
    if (!confirmed) {
      return;
    }

    button.disabled = true;
    setStatus(adminStatus, `Deleting user #${userId}.`, "neutral");

    try {
      await apiRequest(`/users/${userId}`, { method: "DELETE" });
      await loadUsers();
    } catch (error) {
      setStatus(adminStatus, error.message, "error");
      button.disabled = false;
    }
  });

  refreshButton.addEventListener("click", () => {
    loadUsers();
  });

  logoutButton.addEventListener("click", () => {
    apiRequest("/logout", { method: "POST" }).finally(() => {
      clearSession();
      window.location.href = "/";
    });
  });

  loadUsers();
}
