from fastapi import APIRouter, Depends
from app.deps.auth import require_role

router = APIRouter(tags=["admin"])

@router.post("/ingest")
def ingest_docs(current_user = Depends(require_role("admin"))):
    return {
        "message": f"ingest allowed for admin user: {current_user.username}"
    }