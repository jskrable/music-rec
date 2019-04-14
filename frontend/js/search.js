// search.js

search_terms = []

$(function() {
    $('#search-btn').click((e) => {
        var value = $('#search-terms-in').val();
        search_terms.push(value.toLowerCase());
        var close_id = "term-" + value;
        if (value != "") {
            $('#search-terms-div').append(
                '<div class="m-1 d-inline-block alert alert-secondary alert-dismissible fade show" role="alert">' +
                '<strong>' + value + '</strong>' +
                '<button id="' + close_id + '" type="button" class="close" data-dismiss="alert" onclick="closeTerm(this)">' +
                '<span aria-hidden="true">&times;</span>' +
                '</button>' +
                '</div>'
            );
            $('#search-terms-in').val('');
        }
        $('#search-terms-in').focus();
        // Call search function
        filterFileMenu();
    });

    // Include enters
    $('#search-terms-in').keypress((e) => {
        if (e.which == 13) {
            $('#search-btn').click();
        }
    });

});