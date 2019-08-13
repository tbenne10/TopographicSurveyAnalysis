<?php
//Retrieve.php
//Enter a student id to return the drawn pictures and responses. 

$ID = filter_input(INPUT_POST, id);

if(!empty($ret)){
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
          $res = mysqli_query($conn,"SELECT FIRSTNAME,LASTNAME, URL1, URL2, URL3, URL4, URL7, URL8, 
          URL9, URL17, URL18, Q2W, Q7W, Q8W, Q9W, Q10W1, Q10W2, Q13W, Q14W1, Q14W2, Q15W, 
          Q17W, Q18W, Q5R, Q6R, Q11R, Q12R, Q16RAB, Q16RAD, Q16RBC, Q16RCD, Q16RBD, DATE, TIMESTAMP
          FROM Example_table WHERE ID = '$ID'");
          $result = mysqli_fetch_array($res);
          echo "Name: ";
          echo $result['FIRSTNAME'];
          echo " ";
          echo $result['LASTNAME'];
          echo "<br>";
          echo "Date submitted: ";
          echo $result['TIMESTAMP'];
          echo "Question 1: <br>";
          echo '<img src="', $result['URL1'], '">';
          echo "<br>";
          echo "Question 2: <br>";
          echo '<img src="', $result['URL2'], '">';
          echo " -Response: ";
          echo $result['Q2W'];
          echo "<br>";
          echo "Question 3: <br>";
          echo '<img src="', $result['URL3'], '">';
          echo "<br>";
          echo "Question 4: <br>";
          echo '<img src="', $result['URL4'], '">';
          echo "<br>";
          echo "Question 5: ";
          echo $result['Q5R'];
          echo "<br>";
          echo "Question 6: ";
          echo $result['Q6R'];
          echo "<br>";
          echo "Question 7: <br>";
          echo '<img src="', $result['URL7'], '">';
          echo " -Response: ";
          echo $result['Q7W'];
          echo "<br>";
          echo "Question 8: <br>";
          echo '<img src="', $result['URL8'], '">';
          echo " -Response: ";
          echo $result['Q8W'];
          echo "<br>";
          echo "Question 9: <br>";
          echo '<img src="', $result['URL9'], '">';
          echo "<br>";
          echo " -Response: ";
          echo $result['Q9W'];
          echo "<br>";
          echo "Question 10: <br>";
          echo " Steeper Slope: ";
          echo $result['Q10W1'];
          echo "<br>";
          echo " Vertical Distance: ";
          echo $result['Q10W2'];
          echo "<br>";
          echo "Question 10: ";
          echo $result['Q11R'];
          echo "<br>";
          echo "Question 12: ";
          echo $result['Q11R'];
          echo "<br>";
          echo "Question 13: ";
          echo $result['Q13W'];
          echo "<br>";
          echo "Question 14: <br>";
          echo " Steeper Slope: ";
          echo $result['Q14W1'];
          echo "<br>";
          echo " Why?: ";
          echo $result['Q14W2'];
          echo "<br>";
          echo "Question 15: ";
          echo $result['Q15W'];
          echo "<br>";
          echo "Question 16: <br>";
          echo "A and B: ";
          echo $result['Q16RAB'];
          echo "<br>";
          echo "A and D: ";
          echo $result['Q16RAD'];
          echo "<br>";
          echo "B and C: ";
          echo $result['Q16RBC'];
          echo "<br>";
          echo "C and D: ";
          echo $result['Q16RCD'];
          echo "<br>";
          echo "B and D: ";
          echo $result['Q16RBD'];
          echo "<br>";
          echo "Question 17: <br>";
          echo '<img src="', $result['URL17'], '">';
          echo "<br>";
          echo " -Response: ";
          echo $result['Q17W'];
          echo "<br>";
          echo "Question 18: <br>";
          echo '<img src="', $result['URL18'], '">';
          echo "<br>";
          echo " -Response: ";
          echo $result['Q18W'];
          echo "<br>";
          $conn->close();
        }
}else{
echo "Please enter an ID";
die();
}
?>