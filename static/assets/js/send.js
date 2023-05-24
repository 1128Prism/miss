const algorithm = document.getElementById('algorithm-one');
const choice = document.getElementById('choice');

function chooseAlgorithm() {
    $.ajax({
         type: "GET",
         url: "/liveExperience/upload_success",
         dataType: "json",
         data:{"algorithm": algorithm.value},
    });
}

choice.addEventListener('click', chooseAlgorithm);