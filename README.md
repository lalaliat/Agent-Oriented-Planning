# Agent-Oriented Planning in Multi-Agent Systems

This repository contains the code for **AOP**, a novel framework for agent-oriented planning in multi-agent systems.

## Abstract

Through the collaboration of multiple LLM-empowered agents possessing diverse expertise and tools, multi-agent systems achieve impressive progress in solving real-world problems. Given the user queries, the meta-agents, serving as the brain within multi-agent systems, are required to decompose the queries into multiple sub-tasks that can be allocated to suitable agents capable of solving them, so-called agent-oriented planning. In this study, we identify three critical design principles of agent-oriented planning, including solvability, completeness, and non-redundancy, to ensure that each sub-task can be effectively resolved, resulting in satisfactory responses to user queries. These principles further inspire us to propose AOP, a novel framework for agent-oriented planning in multi-agent systems, leveraging a fast task decomposition and allocation process followed by an effective and efficient evaluation via a reward model. According to the evaluation results, the meta-agent is also responsible for promptly making necessary adjustments to sub-tasks and scheduling. Besides, we integrate a feedback loop into AOP to further enhance the effectiveness and robustness of such a problem-solving process. Extensive experiments demonstrate the advancement of AOP in solving real-world problems compared to both single-agent systems and existing planning strategies for multi-agent systems. 

If you find our work helpful for your research or application, please cite our papers.

[Agent-Oriented Planning in Multi-Agent Systems](https://openreview.net/forum?id=EqcLAU6gyU&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FConference%2FAuthors%23your-submissions))

```
@inproceedings{li2025agentoriented,
    title = {Agent-Oriented Planning in Multi-Agent Systems},
    author = {Ao Li and 
              Yuexiang Xie and 
              Songze Li and 
              Fugee Tsung and 
              Bolin Ding and 
              Yaliang Li},
    booktitle = {The Thirteenth International Conference on Learning Representations},
    year = {2025},
    url = {https://openreview.net/forum?id=EqcLAU6gyU}
}
```
