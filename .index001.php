<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
  <title>Customize Plot</title>
</head>
<body>
  <div class="container">
    <h1>Customize Plot</h1>
    <form method="POST">
      <div class="form-group">
        <label for="ticker">Ticker:</label>
        <input type="text" class="form-control" id="ticker" name="ticker" required>
      </div>
      <div class="form-group">
        <label for="color">Color:</label>
        <input type="text" class="form-control" id="color" name="color" required>
      </div>
      <div class="form-group">
        <label for="line_style">Line Style:</label>
        <input type="text" class="form-control" id="line_style" name="line_style" required>
      </div>
      <button type="submit" class="btn btn-primary">Submit</button>
    </form>
    {% if plot_div %}
      {{ plot_div|safe }}
    {% endif %}
  </div>
</body>
</html>
