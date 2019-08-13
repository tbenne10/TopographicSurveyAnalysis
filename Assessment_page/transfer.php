<?php
$ID = filter_input(INPUT_POST, id);

$URL = filter_input(INPUT_POST, URL1); 

if(!empty($ID)){
$host = "";
$dbusername = "";
$dbpassword = "";
$dbname = ""; 

//create conn
$conn = new mysqli ($host, $dbusername, $dbpassword, $dbname);
        if(mysqli_connect_error()){
        die('Connect error ('.mysqli_connect_errno().') '.mysqli_connect_error());
        }
        else{
                $sql = ("INSERT INTO Example_table (ID, URL1, TIMESTAMP)
                values('$ID', '$URL1', NULL)");
                  if ($conn->query($sql)){
                    echo "New record is inserted sucessfully";
                    }
                    else{
                     echo "Error: ". $sql ."". $conn->error;
                    }
          $conn->close();
        }
}else{
echo "Please enter your ID.";
die();
}
?>