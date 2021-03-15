
__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

from src.models import simple_multirotor


def main():

    # estimate simple multirotor drag model
    simple_multirotor.estimate_model("logs/2021-03-11/13_06_40.ulg")

    # preparing needed data
    # angular_accel_df = pandas_from_topic(
    #     ulog, ["vehicle_angular_acceleration"])

    # tecs_df = pandas_from_topic(
    #     ulog, ["airspeed_validated"])

    # airspeed_df = tecs_df[["timestamp", "true_airspeed_m_s"]]

    # print(angular_accel_df)
    # print(tecs_df)
    # print(airspeed_df)

    # ulog_data = get_log_data("logs/2021-03-08/13_25_28.ulg")
    # print(ulog_data)

    return


if __name__ == "__main__":
    main()
