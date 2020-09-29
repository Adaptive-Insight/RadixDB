#https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Structural_Time_Series_Modeling_Case_Studies_Atmospheric_CO2_and_Electricity_Demand.ipynb
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd
from tensorflow_probability import sts
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import collections

num_variational_steps = 200
num_variational_steps = int(num_variational_steps)
optimizer = tf.optimizers.Adam(learning_rate=.1)

def build_model(observed_time_series):
  trend = sts.LocalLinearTrend(observed_time_series=observed_time_series)
  seasonal = tfp.sts.Seasonal(
      num_seasons=12, observed_time_series=observed_time_series)
  model = sts.Sum([trend, seasonal], observed_time_series=observed_time_series)
  return model

# Using fit_surrogate_posterior to build and optimize the variational loss function.
@tf.function(experimental_compile=True)
def train(co2_model, co2_by_month_training_data, variational_posteriors):
  elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn=co2_model.joint_log_prob(
        observed_time_series=co2_by_month_training_data),
    surrogate_posterior=variational_posteriors,
    optimizer=optimizer,
    num_steps=num_variational_steps)
  return elbo_loss_curve

def plot_forecast(x, y,
                  forecast_mean, forecast_scale, forecast_samples,
                  title, x_locator=None, x_formatter=None):
  """Plot a forecast distribution against the 'true' time series."""
  colors = sns.color_palette()
  c1, c2 = colors[0], colors[1]
  fig = plt.figure(figsize=(12, 6))
  ax = fig.add_subplot(1, 1, 1)

  num_steps = len(y)
  num_steps_forecast = forecast_mean.shape[-1]
  num_steps_train = num_steps - num_steps_forecast


  ax.plot(x, y, lw=2, color=c1, label='ground truth')

  forecast_steps = np.arange(
      x[num_steps_train],
      x[num_steps_train]+num_steps_forecast,
      dtype=x.dtype)

  ax.plot(forecast_steps, forecast_samples.T, lw=1, color=c2, alpha=0.1)

  ax.plot(forecast_steps, forecast_mean, lw=2, ls='--', color=c2,
           label='forecast')
  ax.fill_between(forecast_steps,
                   forecast_mean-2*forecast_scale,
                   forecast_mean+2*forecast_scale, color=c2, alpha=0.2)

  ymin, ymax = min(np.min(forecast_samples), np.min(y)), max(np.max(forecast_samples), np.max(y))
  yrange = ymax-ymin
  ax.set_ylim([ymin - yrange*0.1, ymax + yrange*0.1])
  ax.set_title("{}".format(title))
  ax.legend()

  if x_locator is not None:
    ax.xaxis.set_major_locator(x_locator)
    ax.xaxis.set_major_formatter(x_formatter)
    fig.autofmt_xdate()

  return fig, ax

def run(df):
    co2_by_month = df['data'].to_numpy() #np.load('./co2_by_month.npy')
    num_forecast_steps = 12 * 10 # Forecast the final ten years, given previous data
    co2_by_month_training_data = co2_by_month[:-num_forecast_steps]
    co2_dates = df['date'].values.astype('datetime64[M]')

    co2_model = build_model(co2_by_month_training_data)
    # Build the variational surrogate posteriors `qs`.
    variational_posteriors = tfp.sts.build_factored_surrogate_posterior(
        model=co2_model)

    num_variational_steps = 200
    num_variational_steps = int(num_variational_steps)
    optimizer = tf.optimizers.Adam(learning_rate=.1)

    elbo_loss_curve = train(co2_model,  co2_by_month_training_data, variational_posteriors)

    q_samples_co2_ = variational_posteriors.sample(50)
    co2_forecast_dist = tfp.sts.forecast(
        co2_model,
        observed_time_series=co2_by_month_training_data,
        parameter_samples=q_samples_co2_,
        num_steps_forecast=num_forecast_steps)

    num_samples=10
    co2_forecast_mean, co2_forecast_scale, co2_forecast_samples = (
        co2_forecast_dist.mean().numpy()[..., 0],
        co2_forecast_dist.stddev().numpy()[..., 0],
        co2_forecast_dist.sample(num_samples).numpy()[..., 0])

    co2_loc = mdates.YearLocator(3)
    co2_fmt = mdates.DateFormatter('%Y')
    fig, ax = plot_forecast(
        co2_dates, co2_by_month,
        co2_forecast_mean, co2_forecast_scale, co2_forecast_samples,
        x_locator=co2_loc,
        x_formatter=co2_fmt,
        title="Atmospheric CO2 forecast")
    ax.axvline(co2_dates[-num_forecast_steps], linestyle="--")
    ax.legend(loc="upper left")
    ax.set_ylabel("Atmospheric CO2 concentration (ppm)")
    ax.set_xlabel("Year")
    fig.autofmt_xdate()

run(_df)
