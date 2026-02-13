// 输入区域配合sphinx_togglebutton 折叠
document.addEventListener("DOMContentLoaded", function() {
    document.querySelectorAll(".cell_input").forEach(function(cell) {
        cell.classList.add("hide_input");
    });
});
