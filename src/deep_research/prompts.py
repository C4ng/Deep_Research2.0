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
After each round, a separate reflection step assesses your progress and
decides whether more research is needed. If you are called again after
reflection, it means gaps were identified — use the provided guidance
to search for what is missing.
</task>

<research_topic>
{topic}
</research_topic>

<instructions>
1. Read the topic carefully — what specific information is needed?
2. Start with broader searches to understand the landscape.
3. Follow up with narrower, targeted searches to fill gaps.
4. Prefer primary sources (official docs, papers, .gov/.edu) over aggregators.
5. If reflection guidance is provided, act on it:
   - Search for the gaps listed in "Missing information."
   - Use the "Suggested next queries" as starting points.
   - Avoid re-searching topics listed under "Already covered."
</instructions>

<limits>
- Per round: up to {max_searches_per_round} search calls.
- Do not produce a final summary — a downstream step handles synthesis.
</limits>"""


final_report_prompt = """\
Create a comprehensive, well-structured research report based on the
findings below.

<research_brief>
{brief}
</research_brief>

<findings>
{notes}
</findings>

Today's date is {date}.

Requirements:
1. Structure with markdown headings (# title, ## sections, ### subsections).
2. Include specific facts, data, and insights from the findings.
3. Reference sources inline using [Title](URL) format.
4. Be thorough — each section should be substantive, not a brief mention.
5. End with a ### Sources section listing all referenced URLs.
6. Write in the same language as the original user query.

Citation rules:
- Assign each unique URL a sequential citation number [1], [2], [3]...
- Use inline citations: "According to [1], ..."
- List all sources at the end:
  [1] Source Title: URL
  [2] Source Title: URL

Do not refer to yourself or comment on the writing process."""


reflection_prompt = """\
You are assessing the progress of a research task. Today's date is {date}.

<research_topic>
{research_topic}
</research_topic>

<findings>
{findings}
</findings>

<prior_context>
{accumulated_context}
</prior_context>

<instructions>
1. Compare the findings against the research topic — what has been answered?
2. If prior context is provided, assess whether gaps from the last round have been filled by new findings.
3. Identify specific gaps that remain unanswered.
4. Note any contradictions between sources (including against prior findings).
5. Decide whether further searching would be productive.
</instructions>

<field_criteria>
key_findings:
- Include specific facts and data points discovered this round.
- Also capture strategic observations: connections between sources, unexpected scope, quality signals, patterns that inform what to search next.
- These accumulate across rounds — be concrete so future rounds know what's covered.

knowledge_state:
- "insufficient": Core research questions are unanswered, or fewer than 2 supporting sources found.
- "partial": Some questions answered but notable gaps remain.
- "sufficient": All research questions addressed with supporting sources.

should_continue:
- false: Last searches returned mostly overlapping information, topic is too niche for web search, or remaining gaps require expertise search cannot provide.
- true: Concrete gaps exist that targeted queries could fill.

missing_info:
- Be specific and actionable (e.g. "no data on 2024 market share figures"), not vague ("need more information").

contradictions:
- Cite which sources disagree and on what specific point.

next_queries:
- Target the gaps listed in missing_info. Do not repeat prior searches.
</field_criteria>"""


compress_research_prompt = """\
Compress raw research findings into a concise synthesis for a report-writing agent. \
Today's date is {date}.

<research_topic>
{research_topic}
</research_topic>

<raw_findings>
{tool_results}
</raw_findings>

<instructions>
1. Preserve all citations and source URLs — the report needs them.
2. Deduplicate overlapping information across sources.
3. Keep specific facts, data points, statistics, and direct quotes.
4. Remove boilerplate, navigation text, and irrelevant content.
5. Group related information together by subtopic.
6. Target roughly 30% of the original length.
</instructions>"""


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
