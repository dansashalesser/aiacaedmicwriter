# Academic Paper Research Agent - Usage Guide

## Quick Start

The system has been fully strengthened with validation, retry logic, and similarity checking across all 6 phases. You can now run the complete pipeline with a single command.

### Running the System

```bash
cd /Users/dansasha/Documents/aiacaedmicwriter
python3 backend/main.py
```

This will run the complete pipeline with the default research query.

## Customizing Your Research

### Option 1: Edit the Configuration in main.py

Open [backend/main.py](backend/main.py) and modify the configuration section (lines 27-44):

```python
# YOUR RESEARCH INTEREST - Edit this to your topic
research_query = "Your research interest here"

# OPTIONAL: Customize these parameters
num_proposals = 5          # Number of proposals to generate
max_topics = 5            # Maximum topics to segment into
papers_per_topic = 5      # Papers to fetch per topic
papers_for_query = 10     # Papers for original query

# OPTIONAL: Add filters
# min_citation_count = 10
# year_range = (2019, 2024)
# open_access_only = True
```

### Option 2: Run the graph.py Directly

For more control, you can modify and run [backend/graph_app/graph.py](backend/graph_app/graph.py):

```bash
cd /Users/dansasha/Documents/aiacaedmicwriter
python3 backend/graph_app/graph.py
```

## What the System Does

### Phase-by-Phase Pipeline:

1. **Topic Segmentation**: Breaks your research interest into 3-5 focused topics
2. **Literature Search**: Searches Semantic Scholar for papers (parallel searches for each topic)
3. **Knowledge Graph Analysis**:
   - Identifies **intellectual gaps** (not just "no one has studied X+Y")
   - Extracts untested assumptions and contradictions
   - Calculates gap quality score (0-1, where 1 = excellent intellectual gaps)
4. **Proposal Generation**:
   - Generates 5 research proposals
   - **Validates** each proposal (11 validators)
   - **Retries** up to 3 times with targeted feedback if validation fails
   - **Checks similarity** to prevent redundant proposals (75% threshold)
   - **Rejects** proposals that fail all validation attempts

### Output Files

All results are saved to markdown reports:

- **Knowledge Graph**: `backend/graph_app/agents/knowledge_graph_builder/results/`
- **Paper Proposals**: `backend/graph_app/agents/paper_proposal_generator/results/`
- **Papers JSON**: `backend/graph_app/utils/semantic-scholar/results/`

## System Capabilities

### Validation Layers (Phase 3 + 6)

The system validates:
- ✅ **Research Questions**: Specific, answerable, with clear variables
- ✅ **Gaps**: Intellectual (untested assumptions) not coverage-based
- ✅ **Hypotheses**: Quantified predictions with mechanisms and alternatives
- ✅ **Contributions**: Specific findings, not boilerplate
- ✅ **Variables**: Operationalized with measurement details
- ✅ **Risks**: Study-specific, not generic ("data quality issues")
- ✅ **Timeline**: Realistic and internally consistent (12-104 weeks)
- ✅ **Outlets**: Specific journals with quantified rationales

### Retry Logic with Feedback (Phase 4)

When proposals fail validation:
1. Error messages are captured
2. LLM receives **targeted feedback** on specific issues
3. System retries up to **3 times**
4. Each attempt addresses previous errors
5. Proposals that still fail after 3 attempts are **rejected and logged**

### Similarity Checking (Phase 5)

To prevent redundant proposals:
- Uses **sentence-transformers** (all-MiniLM-L6-v2 model)
- Computes **semantic similarity** between proposals
- Rejects proposals >75% similar to existing ones
- Provides **differentiation feedback** when proposals are too similar
- Calculates **average pairwise similarity** (target: <65%)

## Expected Results

After running the system, you should see:

- **Gap Quality Score**: >0.70 (intellectual gaps, not coverage)
- **Validation Pass Rate**: 60-80% of proposals pass within 3 retries
- **Rejection Rate**: <20% rejected after max retries
- **Diversity**: Average pairwise similarity <65%
- **Specificity**: >90% of proposals contain numeric predictions
- **Citations**: Average >5 citations per proposal

## Example Output

```
================================================================================
ACADEMIC PAPER RESEARCH AGENT - STRENGTHENED PROPOSAL GENERATION
================================================================================

📚 Research Interest: Using social media data to improve natural disaster prediction
📊 Will generate: 5 proposals
🔍 Search strategy: 5 topics, 5 papers/topic + 10 for main query

================================================================================

[Stage 1] Segmenting research interest into topics...
   ✓ Generated 5 topics

[Stage 2] Searching Semantic Scholar...
   ✓ Found 45 unique papers

[Stage 3] Building knowledge graph...
   ✓ Analyzed 3 clusters
   ✓ Gap quality score: 0.82

[Stage 4] Generating proposals...
   • Proposal 1: Attempt 1...
      ✓ Validation passed
      ✓ Proposal accepted
   • Proposal 2: Attempt 1...
      ✗ Validation failed: Hypothesis too short
      → Retrying with feedback...
      ✓ Validation passed on attempt 2
      ✓ Proposal accepted
   ✓ Generated 5 valid proposals (0 rejected)
   ℹ Average pairwise similarity: 58% (target: <65%)

================================================================================
✅ PIPELINE COMPLETE - RESULTS SUMMARY
================================================================================

📊 Knowledge Graph Analysis:
   ✓ Gap Quality Score: 0.82 (Excellent - intellectual)

📝 Research Proposals:
   ✓ Generated: 5 valid proposals

   Proposal Titles:
   1. 🟡 Twitter Sentiment Analysis for Real-Time Hurricane Evacuation Prediction
   2. 🟢 Testing the Signal-to-Noise Hypothesis in Disaster Social Media
   3. 🔴 Machine Learning Architectures for Multi-Modal Disaster Forecasting
   4. 🟡 Temporal Dynamics of Social Media Information Cascades During Emergencies
   5. 🟢 Bridging the Digital Divide: Social Media Access and Evacuation Behavior

📁 All results saved to markdown reports
================================================================================
```

## Troubleshooting

### If you get import errors:
```bash
pip install -r requirements.txt
```

### If sentence-transformers is not installed (for similarity checking):
```bash
pip install sentence-transformers scikit-learn
```

### If API calls fail:
- Check that you have a valid OpenAI API key in `.env`
- Check Semantic Scholar API rate limits (they may throttle requests)

### If proposals keep getting rejected:
- This is expected! The system is strict about quality
- Check the `rejected_proposals` output to see why
- The validation feedback is designed to help the LLM improve
- If all proposals fail, you may need to adjust the research query to be more focused

## Development

### Running Individual Tests

Test each phase separately:

```bash
python3 test_phase1.py  # Gap Analysis Reframing
python3 test_phase3.py  # Schema Validators
python3 test_phase4.py  # Retry Logic
python3 test_phase5_structure.py  # Similarity (without model)
python3 test_phase5.py  # Similarity (with model - requires dependencies)
python3 test_phase6.py  # Remaining Validators
```

### Key Files

- **Main Pipeline**: [backend/main.py](backend/main.py)
- **Graph Definition**: [backend/graph_app/graph.py](backend/graph_app/graph.py)
- **Proposal Generator**: [backend/graph_app/agents/paper_proposal_generator/paper_proposal_generator.py](backend/graph_app/agents/paper_proposal_generator/paper_proposal_generator.py)
- **Validators**: [backend/graph_app/agents/paper_proposal_generator/validators.py](backend/graph_app/agents/paper_proposal_generator/validators.py)
- **Similarity Checker**: [backend/graph_app/agents/paper_proposal_generator/similarity.py](backend/graph_app/agents/paper_proposal_generator/similarity.py)
- **Knowledge Graph**: [backend/graph_app/agents/knowledge_graph_builder/knowledge_graph_builder.py](backend/graph_app/agents/knowledge_graph_builder/knowledge_graph_builder.py)

## What's New (All 6 Phases Complete)

### Phase 1: Gap Analysis Reframing ✅
- Reframed from coverage gaps to intellectual gaps
- Added untested assumptions and methodological gaps
- Gap quality scoring

### Phase 2: Prompt Hardening ✅
- Lowered temperature (0.7→0.2) for specificity
- Added explicit anti-patterns throughout prompts
- Strengthened requirements with good/bad examples

### Phase 3: Schema Validators ✅
- Created comprehensive validators module
- Validates: gaps, hypotheses, contributions, variables, research questions
- Integrated into Pydantic schema with `@model_validator`

### Phase 4: Retry Logic with Feedback ✅
- Catches validation errors
- Provides targeted feedback to LLM
- Retries up to 3 times per proposal
- Tracks rejected proposals with error logs

### Phase 5: Similarity Deduplication ✅
- Semantic similarity checking (75% threshold)
- Prevents redundant proposals
- Generates differentiation feedback
- Tracks average pairwise similarity

### Phase 6: Remaining Validators ✅
- Validates risks (rejects generic risks)
- Validates timeline (checks consistency and realism)
- Validates outlets (requires specific journals with quantification)

---

**You're ready to generate scientifically rigorous research proposals!** 🎉
