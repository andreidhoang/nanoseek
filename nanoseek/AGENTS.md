# Where scaling-laws-lab Is Genuinely Superior
I'll be honest — the scaling-laws-lab plan is exceptionally well-designed for its target. Three things it does better than our current plan:

1. Scaling law prediction is a killer differentiator. If you can actually fit L(N,D) = E + A/N^α + B/D^β from 15 small runs and predict 7B loss within 2%, that's a falsifiable, unmistakable signal. Our plan doesn't have this. The curriculum document correctly identifies this as the single most valuable skill for pretraining teams — the ability to predict before spending $10M.

2. Training stability/dynamics coverage is deeper. Our curriculum treats stability as something you encounter while building. The scaling-laws-lab treats it as a first-class pillar: spike injection, spike detection, logit softcapping, QK-norm, z-loss, SPAM optimizer, muP transfer. These are the exact tools Anthropic's on-call engineers use during production training runs. Our plan mentions MFU and profiling but doesn't systematically cover the stability playbook.

3. Observability as a deliverable, not an afterthought. Real-time dashboards, automated spike detection, MFU regression alerts — this is what the Anthropic posting literally describes as "build and maintain production logging, monitoring dashboards." Our plan has profiling (E7), but not a full observability stack.