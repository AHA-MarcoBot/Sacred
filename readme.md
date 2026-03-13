Thank you for your interest and recognition of our work! Here's code for our inter-flow context-aware Application Fingerprinting defense Sacred.

To run the test: 

1. Convert [Jiang2021](https://github.com/jmhIcoding/fgnet) or [CrossNet2021](https://github.com/SecTeamPolaris/ProGraph) or any other desired dataset to *.npz ormat and put into Sacred/sacred_dataset;

2. Use defense flow traces to create defense pool dataset in the same directory. A default 50*10 example dataset is provided.

3. Run train_sacred.py for evaluation under default closed-world scenario. 
