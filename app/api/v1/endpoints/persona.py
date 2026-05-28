from fastapi import APIRouter, HTTPException, status, Depends
from loguru import logger
from typing import List
import uuid
from sqlalchemy import select

from app.schemas.persona import PersonaCreate, PersonaInDB, PersonaUpdate, DerivedPersonaSuggestion
from app.models.persona import Persona
from app.core.database import get_db, AsyncSession
from app.core.security import get_current_user
from app.core.database import User as DBUser
from app.services.derived_persona_service import DerivedPersonaService

router = APIRouter()


@router.post("/derive", response_model=List[DerivedPersonaSuggestion])
async def derive_personas(
    db: AsyncSession = Depends(get_db),
    current_user: DBUser = Depends(get_current_user),
):
    service = DerivedPersonaService(db=db)
    suggestions = await service.derive_personas(user_id=str(current_user.id))
    return suggestions


@router.post("/", response_model=PersonaInDB, status_code=status.HTTP_201_CREATED)
async def create_persona(
    persona_data: PersonaCreate,
    db: AsyncSession = Depends(get_db),
    current_user: DBUser = Depends(get_current_user),
):
    data = persona_data.model_dump()
    data["user_id"] = str(current_user.id)
    db_persona = Persona(**data)

    db.add(db_persona)
    await db.commit()
    await db.refresh(db_persona)

    logger.info(f"Created persona: {db_persona.name} (ID: {db_persona.id}) for user {current_user.id}")
    return PersonaInDB.model_validate(db_persona)


@router.get("/{persona_id}", response_model=PersonaInDB)
async def get_persona(
    persona_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: DBUser = Depends(get_current_user),
):
    db_persona = await db.get(Persona, persona_id)
    if not db_persona or db_persona.user_id != str(current_user.id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Persona not found")
    return PersonaInDB.model_validate(db_persona)


@router.get("/", response_model=List[PersonaInDB])
async def get_all_personas(
    db: AsyncSession = Depends(get_db),
    current_user: DBUser = Depends(get_current_user),
):
    result = await db.execute(
        select(Persona).where(Persona.user_id == str(current_user.id))
    )
    personas = result.scalars().all()
    return [PersonaInDB.model_validate(p) for p in personas]


@router.put("/{persona_id}", response_model=PersonaInDB)
async def update_persona(
    persona_id: uuid.UUID,
    persona_update: PersonaUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: DBUser = Depends(get_current_user),
):
    db_persona = await db.get(Persona, persona_id)
    if not db_persona or db_persona.user_id != str(current_user.id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Persona not found")

    update_data = persona_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_persona, key, value)

    db.add(db_persona)
    await db.commit()
    await db.refresh(db_persona)

    logger.info(f"Updated persona with ID: {persona_id} for user {current_user.id}")
    return PersonaInDB.model_validate(db_persona)


@router.post("/save-derived", response_model=PersonaInDB, status_code=status.HTTP_201_CREATED)
async def save_derived_persona(
    suggestion: DerivedPersonaSuggestion,
    db: AsyncSession = Depends(get_db),
    current_user: DBUser = Depends(get_current_user),
):
    db_persona = Persona(
        name=suggestion.name,
        description=suggestion.description,
        traits=suggestion.traits,
        goals=suggestion.goals,
        user_id=str(current_user.id),
    )
    db.add(db_persona)
    await db.commit()
    await db.refresh(db_persona)
    logger.info(f"Saved derived persona '{db_persona.name}' (ID: {db_persona.id}) for user {current_user.id}")
    return PersonaInDB.model_validate(db_persona)


@router.delete("/{persona_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_persona(
    persona_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: DBUser = Depends(get_current_user),
):
    db_persona = await db.get(Persona, persona_id)
    if not db_persona or db_persona.user_id != str(current_user.id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Persona not found")

    await db.delete(db_persona)
    await db.commit()
    logger.info(f"Deleted persona with ID: {persona_id} for user {current_user.id}")
    return
