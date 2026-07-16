import { getAudienceProfile } from '../productShell/audienceProfiles.ts';
import { useLayoutStore } from '../store/useLayoutStore.ts';

export function useAudienceGuidanceLevel() {
    const audienceMode = useLayoutStore((state) => state.audienceMode);
    return getAudienceProfile(audienceMode).guidanceLevel;
}
