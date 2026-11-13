import logging

import streamlit as st

from pawpal_system import Owner, Pet, Task, Scheduler
from rag_assistant import PetCareAssistant


logger = logging.getLogger(__name__)

st.set_page_config(page_title="PawPal+", page_icon="🐾", layout="centered")

st.title("🐾 PawPal+")

st.markdown(
    """
Welcome to PawPal+. Manage pets and care tasks, view today's schedule, and ask
general pet-care questions with source references.
"""
)

with st.expander("Scenario", expanded=True):
    st.markdown(
        """
**PawPal+** helps a pet owner keep track of their pets and scheduled care tasks.
Its assistant retrieves general care information and shows the references used.

The schedule shows tasks due today in time order; the assistant is not a substitute
for advice from a veterinarian.
"""
    )

with st.expander("What you need to build", expanded=True):
    st.markdown(
        """
PawPal+ lets you:
- Manage an owner profile and multiple pets
- Add and review care tasks for each pet
- View tasks due today in chronological order
- Ask everyday pet-care questions and review retrieved sources
"""
    )

st.divider()

st.subheader("Owner and Pets")
owner_name = st.text_input("Owner name", value="Jordan")
pet_name = st.text_input("Pet name", value="Mochi")
species = st.selectbox("Species", ["dog", "cat", "other"])

# Keep one Owner for the whole session so pets and tasks survive Streamlit reruns.
if "owner" not in st.session_state:
    st.session_state.owner = Owner(owner_name)
owner = st.session_state.owner
owner.name = owner_name

if st.button("Add pet"):
    if any(p.name == pet_name for p in owner.pets):
        st.warning(f"{pet_name} is already on your list.")
    else:
        owner.add_pet(Pet(pet_name, species))

if owner.pets:
    st.write("Your pets:")
    st.table([{"name": p.name, "species": p.species, "tasks": len(p.get_tasks())} for p in owner.pets])
else:
    st.info("No pets yet. Add one above.")

st.markdown("### Tasks")
st.caption("Add care tasks to a pet; tasks due today appear in the schedule below.")

selected_pet_name = st.selectbox("Add task to which pet?", [p.name for p in owner.pets])

col1, col2, col3, col4 = st.columns(4)
with col1:
    task_title = st.text_input("Task title", value="Morning walk")
with col2:
    duration = st.number_input("Duration (minutes)", min_value=1, max_value=240, value=20)
with col3:
    priority = st.selectbox("Priority", ["low", "medium", "high"], index=2)
with col4:
    task_time = st.time_input("Time")

if st.button("Add task"):
    selected_pet = next((p for p in owner.pets if p.name == selected_pet_name), None)
    if selected_pet is None:
        st.warning("Add a pet first, then add tasks to it.")
    else:
        selected_pet.add_task(Task(task_title, task_time.strftime("%H:%M")))
        st.rerun()  # redraw so the "Your pets" task counts above are up to date

# Build table rows from the owner's pets so each task shows which pet it belongs to.
task_rows = [
    {
        "pet": pet.name,
        "task": task.description,
        "time": task.time,
        "frequency": task.frequency,
        "completed": task.completed,
    }
    for pet in owner.pets
    for task in pet.get_tasks()
]

if task_rows:
    st.write("Current tasks:")
    st.table(task_rows)
else:
    st.info("No tasks yet. Add one above.")

st.divider()

st.subheader("Build Schedule")
st.caption("Tasks due today are shown in time order.")

if st.button("Generate schedule"):
    scheduler = Scheduler(owner)
    tasks = scheduler.get_todays_schedule()
    if tasks:
        pet_for_task = {id(task): pet.name for pet in owner.pets for task in pet.get_tasks()}
        st.success(f"Today's Schedule for {owner.name}")
        st.table(
            [
                {"time": task.time, "pet": pet_for_task[id(task)], "task": task.description}
                for task in tasks
            ]
        )
    else:
        st.warning("No tasks are due today. Add a task or check its due date.")

st.divider()

st.subheader("Ask PawPal+")
st.caption(
    "Answers use retrieved references and a local Ollama model when available. "
    "If the model is unavailable, PawPal+ shows retrieved text without claiming "
    "it was AI-generated. This is general information, not veterinary advice."
)
pet_care_question = st.text_area(
    "What would you like to know about everyday pet care?",
    max_chars=500,
    key="pet_care_question",
)

if st.button("Ask question"):
    try:
        with st.spinner("Searching pet-care references..."):
            response = PetCareAssistant().ask(pet_care_question)
        if response.warning:
            st.warning(response.warning)
        if response.safety_redirect or response.insufficient:
            st.info(response.answer)
        else:
            st.write(response.answer)
        st.caption(f"Generation method: {response.generation_method}")
        if response.sources:
            with st.expander("Sources", expanded=True):
                for source in response.sources:
                    st.markdown(f"**{source.title}**")
                    st.markdown(f"[{source.source_name}]({source.source_url})")
                    st.caption(source.content)
    except ValueError as error:
        st.warning(str(error))
    except Exception as error:
        logger.error("Pet-care assistant failed (%s).", type(error).__name__)
        st.error("The assistant could not load its references. Please try again later.")
