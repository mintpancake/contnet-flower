# ConTNet Flower
An implementation of ConTNet on the Oxford 17 Category Flower Dataset

## Useful Statistics
- training set mean: `[0.4214, 0.4245, 0.2806]`
- training set std: `[0.2749, 0.2498, 0.2551]`

## Data Preparation
- Train(80%)/Val(10%)/Test(10%) split:
  1. Extract dataset into `data/flowers/raw`
  2. Run `data/flowers/raw/_split.py`
  3. Run `data/flowers/raw/_generate_meta.py`
  - Please double check with me that our splits are identical

## Dev Instructions
- Workflow:
  1. Checkout a new branch from `main` (Do not work on `main`)
  2. Write code on your branch
  3. Pull `main` and merge your branch on `main`
  4. Solve any conflicts
  5. Push your branch
  6. Create a pull request on GitHub
  7. Inform others of your work
- Dependencies: 
  - Follow HKU CS GPU Farm Tutorial page 8
  - See ./env.yaml
  - Update env `conda env update -f env.yaml`
- Clone issue: 
  - See [创建个人访问令牌](https://docs.github.com/cn/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
