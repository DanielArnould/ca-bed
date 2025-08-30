from tasks.med_dg.data import load_data


data = load_data()

for i in range(10):
    report = data[i]['self_repo']
    target = data[i]['target']
    print(report)
    print(target)