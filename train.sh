while :
do
  python lstm.py
  pkill -9 python
  python glove.py
  pkill -9 python
  python elmo.py
  pkill -9 python
done
