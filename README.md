# PawPal+ (Module 2 Project)

You are building **PawPal+**, a Streamlit app that helps a pet owner plan care tasks for their pet.

## Scenario

A busy pet owner needs help staying consistent with pet care. They want an assistant that can:

- Track pet care tasks (walks, feeding, meds, enrichment, grooming, etc.)
- Consider constraints (time available, priority, owner preferences)
- Produce a daily plan and explain why it chose that plan

Your job is to design the system first (UML), then implement the logic in Python, then connect it to the Streamlit UI.

## What you will build

Your final app should:

- Let a user enter basic owner + pet info
- Let a user add/edit tasks (duration + priority at minimum)
- Generate a daily schedule/plan based on constraints and priorities
- Display the plan clearly (and ideally explain the reasoning)
- Include tests for the most important scheduling behaviors

## Getting started
## Implementation summary

PawPal+ keeps its four core classes in `pawpal_system.py`: Task, Pet, Owner and Scheduler. The application architecture, including the RAG assistant, is in `diagrams/architecture.mmd`.

A Task represents one care activity. It stores a description, a time in 24-hour "HH:MM" format, a frequency ("once", "daily" or "weekly"), a completed flag, and a due date that defaults to today. It can mark itself complete, and its next_occurrence() method returns a new, incomplete copy due one day later for daily tasks or seven days later for weekly tasks. Tasks with any other frequency return nothing.

A Pet stores its name, species, age and its own list of tasks, and it can add a task, remove a task, or return its tasks. An Owner stores a name and a list of pets. It can add or remove pets, and its get_all_tasks() method collects the tasks from every pet into one list.

The Scheduler is connected to a single Owner and reaches tasks through Owner.get_all_tasks(). It sorts tasks by due date and time, filters by pet and completion status, detects exact date-and-time conflicts, and creates the next occurrence of completed daily or weekly tasks. `get_todays_schedule()` returns tasks due today in time order; it accepts an optional date for deterministic use and retains completed tasks so they can be shown in the schedule.

The Streamlit app (`app.py`) manages pets and tasks, generates today's schedule, and provides the pet-care question assistant. The command-line demo (`main.py`) continues to demonstrate scheduling, filtering, conflicts, and recurring tasks. Tests cover both the existing scheduler and the RAG workflow.


### Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Pet-Care Assistant (RAG)

The assistant retrieves relevant passages from `knowledge_base/pet_care.json` before asking a model to answer. Retrieval is a dependency-free keyword-overlap search, not a vector database. Each answer displays the references that were retrieved.

When available, generation uses a local Ollama HTTP service with the `gemma3:1b` model by default. No API key or paid service is used. To enable it, install Ollama, download a model, and run Ollama locally:

```powershell
ollama pull gemma3:1b
ollama serve
python -m streamlit run app.py
```

Set `PAWPAL_OLLAMA_MODEL` to use another locally available model, or `PAWPAL_OLLAMA_URL` to change the local generation endpoint. If the service is unavailable, the app clearly labels its response as an evidence-only extractive fallback; it does not describe that fallback as AI-generated. The fallback uses retrieved text only.

The assistant provides general educational information, not diagnoses or treatment instructions. It redirects high-risk questions toward a veterinarian and declines to answer confidently when retrieval finds no relevant passage. The small curated knowledge base is not a substitute for professional veterinary advice.

### Suggested workflow

1. Read the scenario carefully and identify requirements and edge cases.
2. Draft a UML diagram (classes, attributes, methods, relationships).
3. Convert UML into Python class stubs (no logic yet).
4. Implement scheduling logic in small increments.
5. Add tests to verify key behaviors.
6. Connect your logic to the Streamlit UI in `app.py`.
7. Refine UML so it matches what you actually built.

## 🖥️ Sample Output

Paste a sample of your app's CLI or Streamlit output here so a reader can see what a generated plan looks like:

"Output of python main.py:"
```

🐾 Today's Schedule for Jordan
------------------------------
+-------+----------+--------------------+-----------+----------+
| Time  | Pet      | Task               | Frequency | Status   |
+-------+----------+--------------------+-----------+----------+
| 07:30 | 🐕 Mochi | 🚶 Morning walk    | daily     | ⏳ To do |
| 08:00 | 🐈 Luna  | 🍖 Breakfast       | daily     | ⏳ To do |
| 09:00 | 🐕 Mochi | 🛁 Grooming        | once      | ⏳ To do |
| 09:00 | 🐈 Luna  | 🩺 Vet call        | once      | ⏳ To do |
| 18:00 | 🐕 Mochi | 🎾 Evening play    | daily     | ⏳ To do |
| 20:15 | 🐈 Luna  | 💊 Give medication | once      | ⏳ To do |
+-------+----------+--------------------+-----------+----------+

🔎 Conflict check
-----------------
🚨 WARNING: Conflict on 2026-10-18 at 09:00: Mochi - Grooming, Luna - Vet call

✅ Completed 'Breakfast' (2026-10-18); next one added for 2026-10-19

🐕 Filter: Mochi's tasks (pet_name='Mochi')
-------------------------------------------
+-------+----------+-----------------+-----------+----------+
| Time  | Pet      | Task            | Frequency | Status   |
+-------+----------+-----------------+-----------+----------+
| 18:00 | 🐕 Mochi | 🎾 Evening play | daily     | ⏳ To do |
| 07:30 | 🐕 Mochi | 🚶 Morning walk | daily     | ⏳ To do |
| 09:00 | 🐕 Mochi | 🛁 Grooming     | once      | ⏳ To do |
+-------+----------+-----------------+-----------+----------+

⏳ Filter: incomplete tasks (completed=False)
---------------------------------------------
+-------+----------+--------------------+-----------+----------+
| Time  | Pet      | Task               | Frequency | Status   |
+-------+----------+--------------------+-----------+----------+
| 18:00 | 🐕 Mochi | 🎾 Evening play    | daily     | ⏳ To do |
| 07:30 | 🐕 Mochi | 🚶 Morning walk    | daily     | ⏳ To do |
| 09:00 | 🐕 Mochi | 🛁 Grooming        | once      | ⏳ To do |
| 20:15 | 🐈 Luna  | 💊 Give medication | once      | ⏳ To do |
| 09:00 | 🐈 Luna  | 🩺 Vet call        | once      | ⏳ To do |
| 08:00 | 🐈 Luna  | 🍖 Breakfast       | daily     | ⏳ To do |
+-------+----------+--------------------+-----------+----------+

🐈 Filter: Luna's incomplete tasks (pet_name='Luna', completed=False)
---------------------------------------------------------------------
+-------+---------+--------------------+-----------+----------+
| Time  | Pet     | Task               | Frequency | Status   |
+-------+---------+--------------------+-----------+----------+
| 20:15 | 🐈 Luna | 💊 Give medication | once      | ⏳ To do |
| 09:00 | 🐈 Luna | 🩺 Vet call        | once      | ⏳ To do |
| 08:00 | 🐈 Luna | 🍖 Breakfast       | daily     | ⏳ To do |
+-------+---------+--------------------+-----------+----------+
```

## Formatting Features

## 🎨 CLI Formatting Features

The command-line demo (`main.py`) prints its results as readable, color-coded tables instead of plain text. It uses only Python's standard library, so there is nothing extra to install.

Features:

- Bordered tables: the schedule and every filter result print as aligned tables with the columns Time, Pet, Task, Frequency and Status.
- Pet emojis: 🐕 for dogs, 🐈 for cats and 🐾 for any other species.
- Task emojis: the task description picks the emoji: 🚶 walk, 🍖 feeding or breakfast, 💊 medication, 🎾 play, 🛁 grooming, 🩺 vet, and 📋 for anything else.
- Status indicators: ✅ Done is shown in green and ⏳ To do in yellow.
- Color coding: times are cyan, headings and table headers are bold, conflict warnings are red with a 🚨, and the completed-task message is green.
- Section headings: each part of the demo (schedule, conflict check, filters) has an emoji heading with an underline.

**Functions in `main.py`**

| Function | What it does |
|----------|--------------|
| print_task_table(tasks, pet_for_task) | Builds and prints a bordered table of tasks, with emojis and a colored status. Prints "(no tasks)" if the list is empty. |
| print_heading(title) | Prints a bold section title with an underline. |
| get_pet_emoji(species) | Returns an emoji for the pet's species. |
| get_task_emoji(description) | Returns an emoji by looking for keywords in the task description. |
| colorize(text, code) | Wraps text in an ANSI color code. It does nothing when color is turned off. |
| display_width(text) | Measures how many terminal columns text takes, counting emojis as two, so table columns line up. |
| pad(text, width) | Pads text to a column width using display_width(). |


## Testing PawPal+

```bash
# Run the full test suite:
python -m pytest -q

# Run the deterministic retrieval and grounding evaluation:
python evaluate_rag.py

# Run with coverage:
pytest --cov
```

The evaluation script uses an evidence-only generator, so it makes no model calls. It reports whether each expected knowledge-base item was retrieved and a simple lexical answer/context overlap score. The overlap score is an indicator, not proof that an answer is factually correct.

Sample test output:

```
# Paste your pytest output here
```

## 📐 Smarter Scheduling

> Fill in once you've implemented scheduling logic.

| Feature | Method(s) | Notes |
|---------|-----------|-------|
| Task sorting |Scheduler.sort_by_time()| Sorts all of the owner's tasks by scheduled time in chronological order. Returns a new sorted list without changing the original task lists.|
| Filtering |Scheduler.filter_tasks() | Filters tasks by pet name and/or completion status. Returns a filtered list without deleting or modifying the original tasks.|
| Conflict handling |Scheduler.detect_conflicts() |Reports exact date-and-time matches among unfinished tasks.|
| Today's schedule |Scheduler.get_todays_schedule()| Returns tasks due on the selected date, sorted by time.|

## 📸 Demo Walkthrough

Describe your app in numbered steps so a reader can follow along without watching a video:

1. Open PawPal+: Run the Streamlit application and view the main page. The app provides an interface for managing pets and organizing their care tasks.
2. Review the pets and tasks: View the pet information and the tasks associated with each pet, including task descriptions, scheduled times, and frequencies.
3. Build the schedule: Click the Generate Schedule button to generate the schedule using the scheduling logic in the Scheduler class.
4. Review the sorted schedule: Check that tasks appear in chronological order based on their scheduled times. Each task should show the relevant pet name, task description, frequency, and completion status.
5. Test task filtering: Filter tasks by pet name or completion status to view specific tasks, provided these filtering features have been implemented in the app.
6. Review the results: Confirm that the schedule is readable and that the displayed tasks match the information entered for each pet.


**Screenshot or video** *(optional)*: <!-- Insert a screenshot or link to a demo video here -->


## Testing PawPal+

Run the commands above to see the current test and evaluation results in your environment.

Confidence level:

4/5 stars

I am confident in the reliability of PawPal+ because the basic functionality, task management, and chronological sorting have been implemented. The system organizes tasks for multiple pets and provides a clear schedule that is easy to understand. However, some advanced scheduling features still need to be completed and tested thoroughly. I would give the system four out of five stars because it has a solid foundation, but further testing and improvements are needed to ensure its reliability.

