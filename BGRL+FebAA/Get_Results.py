import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Extraction function
def tflog2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        print(event_acc.Tags())

        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data
path="runs"
print(type(path))
import os
entries = os.listdir(path)
print(entries)
DatasetResults = pd.DataFrame(columns=['DataSet','Results',"StD"])

for e in entries:
    df=tflog2pandas(path+'/'+e)
    print("working on ", e)
    df=df[(df.metric != 'params/lr')&(df.metric != 'params/mm')&(df.metric != 'train/loss')] #delete the mentioned rows
    df.to_csv(path + "/Test_Results.csv")
    counter=0
    total=1
    avrlist=[]
    listofRuns=[]
    for row in df.itertuples(index=True, name='Pandas'):

        print(e,"  ",row.metric, row.value)

        if total==40:
            total=1
            #print(avrlist)
            max_value = max(avrlist)
            listofRuns.append(max_value)
            avrlist = []
            counter=counter+1
            #print(listofRuns)
        elif (row.metric=="accuracy/test_"+str(counter)): #starts from here
            #print(row.metric, row.value)
            avrlist.append(row.value)
            total=total+1

    print(listofRuns)
    import numpy as np
    avg = np.mean(listofRuns)
    print("The average is ", round(avg*100,2))
    values_to_add = {'DataSet': e,'Results': avg*100,"StD": np.std(listofRuns)*100 }
    row_to_add = pd.Series(values_to_add)
    DatasetResults = DatasetResults.append(row_to_add, ignore_index=True)

DatasetResults.to_csv(path+'/Average_Results.csv')