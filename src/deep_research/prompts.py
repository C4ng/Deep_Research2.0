"""Prompt templates for the Deep Research system.

All prompts live here — not scattered in node or tool code.
Grows incrementally as nodes are added.
"""

clarify_prompt = """\
You are preparing to research a topic for the user. Assess whether you
need to ask a clarifying question before starting.

<messages>
{messages}
</messages>

Today's date is {date}.

Guidelines:
1. Ask only when genuinely needed — unclear acronyms, ambiguous scope that
   could lead to very different research directions, or missing context
   that would waste research effort.
2. If the message history shows you have already asked a clarifying
   question, almost never ask another. Only ask again if absolutely
   necessary.
3. Do not ask for information the user already provided.
4. When asking, be concise — one well-structured question, not a list
   of demands. Use markdown formatting for readability.
5. When not asking, provide a brief verification message: acknowledge
   what you understand from the request and confirm you will begin
   research."""


research_brief_prompt = """\
You will be given messages from a user requesting research on a topic.
Create a draft research plan that will be reviewed by the user before
research begins. The plan has two parts: a clear research question and
a recommended approach.

<messages>
{messages}
</messages>

<prior_brief>
{prior_brief}
</prior_brief>

<user_feedback>
{feedback}
</user_feedback>

Today's date is {date}.

If a prior brief and user feedback are provided above, assess whether the
user is approving the brief or requesting changes. If approving, keep the
brief unchanged and set ready_to_proceed to true. If requesting changes,
revise accordingly and set ready_to_proceed to false.

Research question — what to investigate:
1. Capture the user's question with full specificity — include all
   preferences, constraints, and context they provided.
2. If certain dimensions are essential but the user didn't specify them,
   explicitly state they are open-ended. Do not invent requirements.
3. If specific sources should be prioritized, include them.
4. If the query is in a specific language, note to prioritize sources
   in that language.
5. Phrase from the user's perspective — preserve their voice and intent.

Approach — how to investigate:
6. Identify what kind of research this question needs. What are the
   important angles or facets to cover? Should the research prioritize
   breadth (many angles, surface-level) or depth (fewer angles, thorough)?
7. Note any priorities — which aspects matter most, what would make the
   research most useful to the user.
8. This is strategic guidance, not an exact topic list. A downstream
   coordinator will decide the specific decomposition into research tasks.
9. Use bullet points — easy to scan and modify.

Simplicity assessment:
10. Determine if this is a simple question that a single researcher can
    handle without multi-topic decomposition. Examples: "What is the
    latest stable version of React?", "When was company X founded?"
    Set is_simple accordingly."""


researcher_prompt = """\
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
Create a comprehensive research report based on the findings below.

<research_brief>
{brief}
</research_brief>

<findings>
{notes}
</findings>

<research_metadata>
{report_metadata}
</research_metadata>

<source_map>
{source_map}
</source_map>

Today's date is {date}.

<instructions>
1. Structure with markdown headings (# title, ## sections, ### subsections).
2. Write substantive sections — each should contain specific facts, data, and
   analysis from the findings, not brief mentions.
3. Cite sources inline using their [source_id] tags exactly as they appear in
   the findings (e.g., [a1b2c3d4]). Do NOT invent source IDs — only use IDs
   that appear in the findings or source map.
4. When research_metadata lists contradictions, present them within the relevant
   topic section using a clear subheading (e.g., ### Conflicting Evidence) so
   they are visually distinct and scannable. For each contradiction: present
   both sides with their sources, analyze why they may differ (methodology,
   timeframe, source type, scope), and offer the reader guidance on which
   claim has stronger support or what factors to consider. Do not just list
   both sides without analysis.
5. When research_metadata contains gaps, present them within the relevant topic
   section using a clear subheading (e.g., ### Open Questions). The metadata
   labels indicate why each gap exists (not investigated, searched but not found,
   partial coverage) — use these signals to explain what remains unknown and why.
6. Use the coverage/knowledge_state signals in research_metadata to calibrate
   confidence in the main body. Topics with "partial" coverage should use
   hedging language. Topics with "sufficient" coverage can be more assertive.
7. Do NOT write a Sources or References section — it will be added
   programmatically after generation.
</instructions>

Do not refer to yourself or comment on the writing process."""


# TODO(calibration): key_findings field_criteria says "specific facts and data points"
# which may push toward granular detail rather than high-level insights. Observe what
# the LLM produces and calibrate — key findings should be the important discoveries
# that answer the research question, not a laundry list of every data point.
researcher_reflection_prompt = """\
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
1. Compare the findings against the research topic — does the information
   answer what the topic actually asks for, at the level of detail it implies?
   A topic asking about "advancements in 2025" needs key developments, not
   exhaustive metrics for every sub-technology.
2. If prior context is provided, check whether gaps from the last round were
   filled by new findings. If the same gaps persist despite targeted searches,
   they are likely unsearchable — do not keep chasing them.
3. Identify gaps that are within the scope of the assigned topic and
   realistically answerable by web search. Do not invent deeper questions
   beyond what the topic asks for.
4. Note any contradictions between sources (including against prior findings).
5. Decide whether further searching would be productive — prioritize
   stopping with good coverage over exhaustive completeness.
</instructions>

<field_criteria>
key_findings:
- Include specific facts and data points discovered this round.
- Reference sources using their [source_id] tags from the search results
  (e.g., "Market reached $3.77B [a1b2c3d4]").
- Also capture strategic observations: connections between sources, unexpected
  scope, quality signals.
- These accumulate across rounds — be concrete so future rounds know what's covered.

knowledge_state:
- "insufficient": Core questions in the topic are unanswered, fewer than 2
  supporting sources found.
- "partial": Topic is partially answered but major angles are missing.
- "sufficient": The topic is well-covered at the level of detail it asks for.
  Minor niche details missing does NOT prevent "sufficient."

should_continue:
- false: The topic is reasonably well-covered; OR last searches returned mostly
  overlapping information; OR remaining gaps are too niche for web search; OR
  gaps from the previous round persist unfilled (signal they are unsearchable).
- true: Major aspects of the topic are unanswered AND targeted queries are
  likely to help.

missing_info:
- Only gaps that the topic explicitly asks for but the findings don't cover.
- Do NOT list deeper follow-up questions beyond the topic's scope.
- Do NOT list niche details when the topic asks for a general overview.
- Be specific (e.g. "no information on regulatory landscape" not "need more info")
  but stay at the topic's level of detail — not niche sub-questions.

contradictions:
- Cite which sources disagree using their [source_id] tags
  (e.g., "[a1b2c3d4] says X, [e5f6a7b8] says Y").

next_queries:
- Target the gaps listed in missing_info. Do not repeat prior searches.

prior_gaps_filled:
- Count how many gaps listed in prior_context's "Gaps identified last round"
  were substantively answered by this round's findings.
- 0 if there were no prior gaps, or if none were filled.
- A gap is "filled" if the findings now contain a meaningful answer, even if
  partial. A gap is "unfilled" if the search returned nothing relevant.
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
1. Reference each finding by its source ID (e.g., [a1b2c3d4]). Source IDs
   appear in the raw findings as [source_id] tags — preserve them exactly.
   Do NOT invent or modify source IDs.
2. Deduplicate overlapping information across sources.
3. Keep specific facts, data points, statistics, and direct quotes.
4. Remove boilerplate, navigation text, and irrelevant content.
5. Group related information together by subtopic.
6. Target roughly 30% of the original length.
</instructions>"""


coordinator_prompt = """\
You are a research coordinator managing a team of focused researchers. \
Today's date is {date}.

<task>
Read the research brief and dispatch researchers using `dispatch_research`.
Each researcher works independently on its assigned topic — it does not
see other researchers' findings or the full brief.
</task>

<research_brief>
{research_brief}
</research_brief>

<prior_research>
{prior_research}
</prior_research>

<instructions>
1. Read the research brief and decompose it into focused, complementary
   subtopics that together answer the research question. The brief may
   include approach guidance — use it as a starting point, but apply your
   own judgment about what topics will produce the most useful research.
2. For each subtopic, call `dispatch_research(topic, context)`:
   - `topic`: a focused, specific research topic (not the full brief).
   - `context`: brief context explaining why this subtopic matters and
     what angle to investigate. Include relevant constraints from the brief.
3. Make topics complementary, not overlapping — each researcher should
   cover a distinct angle.
4. If prior research results are shown above, read them before dispatching:
   - Do not re-research topics already covered with sufficient knowledge.
   - Target gaps and contradictions identified in the reflection.
   - Look for emergent topics — important angles within the scope of the
     original user query that complement existing topics. Emergent topics
     are missing perspectives at the same level (e.g., a stakeholder nobody
     covered), NOT deeper drill-downs into already-covered areas.
5. Do not produce a final summary — a downstream step handles synthesis.
</instructions>

<limits>
- Soft cap: up to {max_research_topics} subtopics per round.
- Follow-up rounds: dispatch researchers for gaps and emergent topics.
</limits>"""


coordinator_reflection_prompt = """\
You are assessing the completeness of a multi-topic research effort. \
Today's date is {date}.

<research_brief>
{research_brief}
</research_brief>

<research_results>
{research_results}
</research_results>

<instructions>
1. Compare the combined research results against the original brief —
   does the research answer the user's questions at the level of detail
   the brief implies? Do not demand exhaustive coverage of every sub-detail.
2. Note any emergent topics discovered during research — important angles
   within the scope of the original query that complement existing coverage.
   Mention these in the overall assessment. Emergent topics are missing
   perspectives at the same level (e.g., a stakeholder group nobody covered),
   NOT deeper drill-downs into already-covered areas.
3. Identify contradictions *between* different researchers' findings
   (not within a single researcher — those are already flagged).
4. Assess whether remaining gaps are major (worth dispatching a researcher)
   or minor (niche details that web search is unlikely to resolve).
5. Prefer stopping with good coverage over chasing exhaustive completeness.
   Follow-up research should only be dispatched for clearly significant gaps,
   not for granular details.
</instructions>

<field_criteria>
overall_assessment:
- One paragraph summarizing how well the research covers the user's brief.
- Note any emergent topics discovered that complement existing coverage.

cross_topic_contradictions:
- Cite which researchers disagree and on what specific point.
- Only flag genuine conflicts, not differences in scope or emphasis.

coverage_gaps:
- Only major aspects of the brief not addressed by any researcher.
- Do NOT list deeper drill-downs into already-covered topics.
- Do NOT list niche details that researchers flagged as missing — those are
  expected and not worth a new researcher dispatch.
- Do NOT include emergent topics here — note them in overall_assessment instead.
- Each gap should be significant enough to justify dispatching a full researcher.

should_continue:
- true: A major aspect of the brief is unaddressed AND a targeted researcher
  could realistically fill it via web search.
- false: The brief's research questions are reasonably answered; remaining
  gaps are minor, too niche, or already attempted unsuccessfully.

knowledge_state:
- "insufficient": Major research questions from the brief are unanswered.
- "partial": Most questions answered but a significant angle is missing.
- "sufficient": The brief's questions are well-covered at the level of detail
  they ask for. Minor gaps do NOT prevent "sufficient."
</field_criteria>"""


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
