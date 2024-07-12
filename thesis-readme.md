python thesis_plotter_var.py --network renyi --attack 2 --lams 0 0.3 0.6 --vars 0.0 0.5 1.0 1.5


python thesis_subplotter.py --network complete --attack 2 --lams 0 0.3 0.6 0.9
python thesis_subplotter.py --network renyi --attack 2 --lams 0 0.3 0.6 0.9

python thesis_subplotter.py --network h2b1 --attack 2 --lams 0 0.3 0.6 0.9
python thesis_subplotter.py --network h3b1 --attack 2 --lams 0 0.3 0.6 0.9
python thesis_subplotter.py --network h4b1 --attack 2 --lams 0 0.3 0.6 0.9

python thesis_subplotter.py --network renyi --attack 0 --lams 0 0.3 0.6 0.9
python thesis_subplotter.py --network renyi --attack 1 --lams 0 0.3 0.6 0.9

---------------------

python thesis_subplotter.py --network renyi --attack 0 --lams 0 0.3 0.6 0.9
python thesis_subplotter-slides.py --network renyi --attack 1 --lams 0 0.3 0.6 0.9
python thesis_subplotter.py --network renyi --attack 2 --lams 0 0.3 0.6 0.9
