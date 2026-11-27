# Data contract

The build-snapshot command reads JSONL per trajectory with fields:
case_id, task, split (train/validation/test), instruction, video, frame_count,
optional environment_success (0/1), and progress entries {frame,value,known}.
An optional annotation_result is an already-labelled structured record containing
proposed_annotation and its events/action associations.

~~~bash
python -m failatlas build-snapshot --manifest trajectories.jsonl \
 --ontology taxonomy/ontology.json --video-root /path/to/videos --output work/snapshot
~~~

--terminal-only creates terminal T-only rows; it never generates missing A.
Trajectory IDs cannot repeat or appear across splits. Supply the original split
membership: this builder does not randomly re-split anything.

Histories contain at most eight observations ending at the scoring frame.
Model inputs contain only instructions and observed images. T supervision comes
only from the environment endpoint. Unknown fields/categories are masked;
other valid labels in the sample remain usable.
--exhaustive-failure-labels is explicit and applies only to genuinely exhaustive labels.

This creates a NEW snapshot and hashes. Rebuilding it changes snapshot identity.
Historical checkpoint replay requires the matching frozen JSONL plus verified
media/cache mapping; it must not silently accept an edited snapshot.

Raw pool124354 differs from usable full-T pool124246, with108 reserved exclusions.
Structured train/validation/test:23608/2941/2951. Raw web-database exports contain
internal/historical records and require clean export before hosting.
