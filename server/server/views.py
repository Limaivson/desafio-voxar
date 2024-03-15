from django.http import JsonResponse
from .read_folder import ReadFolder

read_folder = ReadFolder()

"""
Preview function to receive a folder path via POST and process the images contained therein.

Parameters:
    - request (HttpRequest): HTTP request object sent by the client.

Returns:
    - JsonResponse: A JSON response indicating the success or failure of processing the folder path.
"""


def receive_path(request):
    # Checks whether the request is of type POST.
    if request.method == 'POST':
        # Gets the folder path sent in the request.
        data = request.POST.get('path')
        # Reads images in the specified folder and processes their categories.
        read = read_folder.read_image_in_folder(data)
        # Checks whether any errors occurred while processing the folder.
        if read:
            # Returns a JSON response indicating the error that occurred.
            return JsonResponse({'message': 'Erro: ' + read}, status=400)
        else:
            # Returns a JSON response indicating successful processing of the folder.
            return JsonResponse({'message': 'Recebido'})
