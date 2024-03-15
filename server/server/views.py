from django.http import JsonResponse
from .read_folder import ReadFolder

read_folder = ReadFolder()


def receive_path(request):
    if request.method == 'POST':
        data = request.POST.get('path')
        read = read_folder.read_image_in_folder(data)
        if read:
            return JsonResponse({'message': 'Erro: ' + read}, status=400)
        else:
            return JsonResponse({'message': 'Recebido'})
        