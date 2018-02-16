import DS1
import DS2_keras
import DS3 as DS3

if __name__ == "__main__":
    print('='*50)
    print('\t\tEstimation of UPDRS')
    DS2_keras.train()  # load data and train ds1 network
    print('=' * 50)
    print('\t\tRecognizing Parkinson disease from voice measurements')
    DS1.train()
    print('=' * 50)
    print('\t\tRecognizing Parkinson disease from draw tests')
    DS3.train()
    print('=' * 50)
