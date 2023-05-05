from datetime import datetime, timedelta
def getTrainMaxSpeed(train_id):
    train_id = str(train_id)
    if train_id[0].isdigit():
        return "80"
    elif train_id[0].isalpha():
        if train_id[0] == "G" or train_id[0] == "C":
            return "350"
        elif train_id[0] == "D":
            return "260"
        elif train_id[0] == "Z" or train_id[0] == "T":
            return "160"
        else:
            return "120"


def convertTimetoMinute(time, day=None):

    """
    Process outliers time datas
    Some arrive and depart time are in fraction of a day, so we need to convert them to minutes
    ex: 0.8884 = 21:30:00
    """

    try:
        time_float = float(time)
        # Convert the fraction of a day to a timedelta object
        delta = timedelta(days=time_float)
        start_of_day = datetime(year=1, month=1, day=1)
        time = start_of_day + delta
    except:
        pass


    if isinstance(time, str):
        try:
            time = datetime.strptime(time, "%H:%M:%S")
        except:
            # Parse time to get the first 2 digits
            time_slipt = time.split(":")[0]
            dif = 24 - int(time_slipt)
            time = time.replace(time_slipt+":", "23:")
            time = datetime.strptime(time, "%H:%M:%S")
            time += timedelta(hours=dif)

    if day == "Day 1" or day == None:
        minutes = time.hour * 60 + time.minute
    elif day == "Day 2":
        minutes = time.hour * 60 + time.minute + 24 * 60
    elif day == "Day 3":
        minutes = time.hour * 60 + time.minute + 24 * 60 * 2
    elif day == "Day 4":
        minutes = time.hour * 60 + time.minute + 24 * 60 * 3

    return minutes