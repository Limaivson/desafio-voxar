from django.http import JsonResponse
from .read_folder import ReadFolder

read_folder = ReadFolder()


def receive_path(request):
    if request.method == 'POST':
        data = request.POST.get('path')
        read_folder.read_image_in_folder(data)
        return JsonResponse({'message': 'Recebido'})
    else:
        return JsonResponse({'error': 'Método não permitido'}, status=405)
