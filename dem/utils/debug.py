
import wandb


def fprint(s):
    print(s)
    with open(f"{wandb.run.dir}/debug.txt", "a+") as fout:
        fout.write(f"{s}\n")