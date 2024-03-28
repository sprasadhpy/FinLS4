def combine_vectors(self, x, y, dim=-1):
    '''
    Function for combining two tensors
    '''
    combined = torch.cat([x, y], dim=dim)
    combined = combined.to(torch.float)
    return combined