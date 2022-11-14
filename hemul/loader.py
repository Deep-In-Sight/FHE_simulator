import hemul

def load():
    if hemul.USE_FPGA:
        import hemul.HEAAN_fpga as he
        print("Using FPGA version HEAAN")
    elif hemul.USE_CUDA:
        import hemul.HEAAN_cuda as he
        print("Using CUDA version HEAAN")
    else:
        import hemul.HEAAN as he
        print("Using CPU version HEAAN")

    return he