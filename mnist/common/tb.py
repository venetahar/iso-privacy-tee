from tensorboard import program


tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', './'])
url = tb.launch()
print(url)
