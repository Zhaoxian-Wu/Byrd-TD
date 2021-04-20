1. Description
Code of paper 'Byzantine-Resilient Decentralized TD Learning with Linear Function Approximation'. 
'robust_td.py' is the main code. 'plotter.py' and 'subplotter.py' are for plotting. Other codes are utilities.

2. Instruction
(1) Check environment requirements: python(>=3.6), numpy, matplotlib, scipy, gym(0.10.5), tensorflow(>=2.0.0)
(2) Install the multiagent env package in this file. cd into root of this file and run: pip install -e multiagent-particle-envs-master
(3) An example to run the code in console: python robust_td.py --network renyi --mc 1 --attack 2 --lam 0. --lr 0.1 --diminish --plot