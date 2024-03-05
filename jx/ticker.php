<html>
<head><meta charset="utf-8" /></head>
<body>
    <?php
     = htmlspecialchars(["ticker"]);
     = "tech/" .  . ".php";

    if (file_exists()) {
        include ;
    } else {
        echo "No plot available for " . ;
    }
    ?>
</body>
</html>
