# Description
Code of paper 'Byzantine-Resilient Decentralized TD Learning with Linear Function Approximation'. ([Arxiv preprint: https://arxiv.org/abs/2009.11146](https://arxiv.org/abs/2009.11146))

'robust_td.py' is the main code. 'plotter.py' and 'subplotter.py' are for plotting. Other codes are utilities.

# Instruction
1. Check environment requirements: python(>=3.6), numpy, matplotlib, scipy, gym(0.10.5), tensorflow(>=2.0.0)
2. Install the multiagent env package in this file. cd into root of this file and run: pip install -e multiagent-particle-envs-master
3. An example to run the code in console: python robust_td.py --network renyi --attack 2 --lam 0. --lr 0.1 --diminish --plot

# Running
## Different Topologies
Figure: MSBE and MCE versus step k under sign flipping attacks in a complete network.
```bash
python robust_td.py --network complete --attack 2 --lam 0. --lr 0.1 --diminish --mc 10
python robust_td.py --network complete --attack 2 --lam 0.3 --lr 0.1 --diminish --mc 10
python robust_td.py --network complete --attack 2 --lam 0.6 --lr 0.1 --diminish --mc 10
python robust_td.py --network complete --attack 2 --lam 0.9 --lr 0.05 --diminish --mc 10
python subplotter.py --network complete --attack 2 --lams 0 0.3 0.6 0.9 --lnk
```

---
Figure: MSBE and MCE versus step k under sign flipping in a Erdos-Renyi network.
```bash
python robust_td.py --network renyi --attack 2 --lam 0. --lr 0.1 --diminish --mc 10
python robust_td.py --network renyi --attack 2 --lam 0.3 --lr 0.1 --diminish --mc 10
python robust_td.py --network renyi --attack 2 --lam 0.6 --lr 0.1 --diminish --mc 10
python robust_td.py --network renyi --attack 2 --lam 0.9 --lr 0.05 --diminish --mc 10
python subplotter.py --network renyi --attack 2 --lams 0 0.3 0.6 0.9 --lnk
```

---
Figure: MSBE and MCE versus step k under sign flipping in a H2B1 network.
```bash
python robust_td.py --network h2b1 --attack 2 --lam 0. --lr 0.1 --diminish --mc 10
python robust_td.py --network h2b1 --attack 2 --lam 0.3 --lr 0.1 --diminish --mc 10
python robust_td.py --network h2b1 --attack 2 --lam 0.6 --lr 0.1 --diminish --mc 10
python robust_td.py --network h2b1 --attack 2 --lam 0.9 --lr 0.05 --diminish --mc 10
python subplotter.py --network h2b1 --attack 2 --lams 0 0.3 0.6 0.9 --lnk
```

---
Figure: MSBE and MCE versus step k under sign flipping in a H3B1 network.
```bash
python robust_td.py --network h3b1 --attack 2 --lam 0. --lr 0.1 --diminish --mc 10
python robust_td.py --network h3b1 --attack 2 --lam 0.3 --lr 0.1 --diminish --mc 10
python robust_td.py --network h3b1 --attack 2 --lam 0.6 --lr 0.1 --diminish --mc 10
python robust_td.py --network h3b1 --attack 2 --lam 0.9 --lr 0.05 --diminish --mc 10
python subplotter.py --network h3b1 --attack 2 --lams 0 0.3 0.6 0.9 --lnk
```
---
Figure: MSBE and MCE versus step k under sign flipping in a H4B1 network.
```bash
python robust_td.py --network h4b1 --attack 2 --lam 0. --lr 0.1 --diminish --mc 10
python robust_td.py --network h4b1 --attack 2 --lam 0.3 --lr 0.1 --diminish --mc 10
python robust_td.py --network h4b1 --attack 2 --lam 0.6 --lr 0.1 --diminish --mc 10
python robust_td.py --network h4b1 --attack 2 --lam 0.9 --lr 0.05 --diminish --mc 10
python subplotter.py --network h4b1 --attack 2 --lams 0 0.3 0.6 0.9 --lnk
```

## Different Byzantine attacks
Figure: MSBE and MCE versus step k under same value in a Erdos-Renyi network.
```bash
python robust_td.py --network renyi --attack 0 --lam 0. --lr 0.1 --diminish --mc 10
python robust_td.py --network renyi --attack 0 --lam 0.3 --lr 0.1 --diminish --mc 10
python robust_td.py --network renyi --attack 0 --lam 0.6 --lr 0.1 --diminish --mc 10
python robust_td.py --network renyi --attack 0 --lam 0.9 --lr 0.05 --diminish --mc 10
python subplotter.py --network renyi --attack 0 --lams 0 0.3 0.6 0.9 --lnk
```

---
Figure: MSBE and MCE versus step k under Gaussian noise in a Erdos-Renyi network.
```bash
python robust_td.py --network renyi --attack 1 --lam 0. --lr 0.1 --diminish --mc 10
python robust_td.py --network renyi --attack 1 --lam 0.3 --lr 0.1 --diminish --mc 10
python robust_td.py --network renyi --attack 1 --lam 0.6 --lr 0.1 --diminish --mc 10
python robust_td.py --network renyi --attack 1 --lam 0.9 --lr 0.05 --diminish --mc 10
python subplotter.py --network renyi --attack 1 --lams 0 0.3 0.6 0.9 --lnk
```

## Different reward variation
Figure: Asymptotic MSBE under sign flipping in a H3B1 network.
```bash
python robust_td.py --network renyi --attack 2 --lam 0. --lr 0.05 --diminish --vars 0 0.5 1.0 1.5
python robust_td.py --network renyi --attack 2 --lam 0.3 --lr 0.05 --diminish --vars 0 0.5 1.0 1.5
python robust_td.py --network renyi --attack 2 --lam 0.6 --lr 0.05 --diminish --vars 0 0.5 1.0 1.5
python robust_td.py --network renyi --attack 2 --lam 0.9 --lr 0.05 --diminish --vars 0 0.5 1.0 1.5
python plotter_var.py --network renyi --attack 2 --lams 0 0.3 0.6 0.9 --vars 0.0 0.5 1.0 1.5
```