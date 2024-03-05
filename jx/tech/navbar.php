<html>
<head><meta charset="utf-8" /></head>
<body>
    <nav>
        <?php
        $files = glob("tech/*.php");
        foreach ($files as $file) {
            $name = basename($file, ".php");
            echo "<a href='$file'>$name</a><br>";
        }
        ?>
    </nav>
</body>
</html>