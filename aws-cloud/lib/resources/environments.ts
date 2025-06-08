export const ENV = {
    DEPLOYMENT_ENV    : process.env.DEPLOYMENT_ENV    ?? 'dev',
    HOSTED_ZONE       : process.env.HOSTED_ZONE       ?? '',
    ACR_ACCESS_KEY    : process.env.ACR_ACCESS_KEY    ?? '',
    ACR_ACCESS_SECRET : process.env.ACR_ACCESS_SECRET ?? '',
} as const;