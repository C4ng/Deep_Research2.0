"""Prompt templates for the Deep Research system.

All prompts live here — not scattered in node or tool code.
Grows incrementally as nodes are added.
"""

research_brief_prompt = """\
You will be given messages from a user requesting research on a topic.
Transform these into a structured research brief that will guide the
research process.

<messages>
{messages}
</messages>

Today's date is {date}.

Guidelines:
1. Extract the core research question and break it into specific sub-questions.
2. Identify key topics and subtopics that need investigation.
3. Preserve all user-specified constraints, preferences, and context.
4. If the user mentions specific sources or domains, include them.
5. Do not invent requirements the user did not state — leave unspecified
   dimensions open rather than assuming.
6. Phrase the brief from the user's perspective.
7. Be specific and detailed — this brief is the sole input the researcher sees."""


research_system_prompt = """\
You are a research assistant investigating a specific topic. Today's date is {date}.

<task>
Use the search tools to gather information about the research topic below.
You can call tools in series or parallel within a tool-calling loop.
</task>

<research_topic>
{topic}
</research_topic>

<instructions>
1. Read the topic carefully — what specific information is needed?
2. Start with broader searches to understand the landscape.
3. After each search, assess: do I have enough? What gaps remain?
4. Follow up with narrower, targeted searches to fill gaps.
5. Stop when you can answer confidently — do not search endlessly.
6. Prefer primary sources (official docs, papers, .gov/.edu) over aggregators.
</instructions>

<limits>
- Simple queries: 2-3 search calls maximum.
- Complex queries: up to 5 search calls maximum.
- Stop immediately when:
  - You can answer the research question comprehensively.
  - You have 3+ relevant sources covering the key aspects.
  - Your last 2 searches returned overlapping information.
</limits>"""


summarize_webpage_prompt = """\
Summarize the raw content of a webpage for use by a downstream research agent. \
Preserve the most important information without losing essential details.

Guidelines:
1. Preserve the main topic or purpose of the webpage.
2. Retain key facts, statistics, and data points.
3. Keep important quotes from credible sources.
4. Maintain chronological order for time-sensitive content.
5. Preserve lists or step-by-step instructions if present.
6. Include relevant dates, names, and locations.
7. Aim for about 25-30% of the original length.

Content types:
- News articles: focus on who, what, when, where, why, how.
- Scientific content: preserve methodology, results, conclusions.
- Opinion pieces: maintain main arguments and supporting points.
- Product pages: keep key features, specifications, selling points.

Today's date is {date}.

<webpage_content>
{webpage_content}
</webpage_content>"""
