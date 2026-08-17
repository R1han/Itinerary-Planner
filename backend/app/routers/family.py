"""Family composition for the current user. PUT replaces the whole set — it is small and edited
as a unit in the UI, so a replace is simpler and less error-prone than per-member CRUD."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..db import get_db
from ..models import FamilyMember, User
from ..repo import owned_query
from ..schemas import FamilyMemberOut, FamilyUpdate

router = APIRouter(prefix="/family", tags=["family"])


@router.get("", response_model=list[FamilyMemberOut])
def get_family(
    current: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> list[FamilyMember]:
    return owned_query(db, FamilyMember, current.id).order_by(FamilyMember.id).all()


@router.put("", response_model=list[FamilyMemberOut])
def replace_family(
    payload: FamilyUpdate,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[FamilyMember]:
    owned_query(db, FamilyMember, current.id).delete()
    for member in payload.members:
        db.add(FamilyMember(user_id=current.id, **member.model_dump()))
    db.commit()
    return owned_query(db, FamilyMember, current.id).order_by(FamilyMember.id).all()
